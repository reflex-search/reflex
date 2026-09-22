//! Latency budget harness.
//!
//! Measures the exact query shapes from the field test on a deterministic
//! ~30 MiB / 2000-file synthetic corpus (see `test_helpers::synthetic_corpus`),
//! twice:
//!
//! - `latency_in_process`: `QueryEngine::search_with_metadata` directly. A fresh
//!   `CacheManager` + `QueryEngine` is built per call, mirroring what the MCP
//!   `search_code` handler does, so the MCP − in-process delta isolates the
//!   transport (JSON-RPC framing, serialisation, freshness check).
//! - `latency_mcp_stdio`: a real `rfx mcp` child process driven over stdio.
//!
//! Both are `#[ignore]`d — run them with `--test-threads=1` (they contend for
//! CPU otherwise and the in-process numbers absorb the MCP child's load):
//!
//! ```text
//! cargo test --release --test latency_budget -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Environment:
//! - `REFLEX_LATENCY_BUDGET=1` — assert medians against [`SHAPES`] budgets.
//! - `REFLEX_LATENCY_JSON=1`   — also print one JSON line per shape.
//!
//! The parity asserts (hit counts vs a word-boundary scan of the generated
//! files) always run, so the harness doubles as a correctness check.

mod test_helpers;

use reflex::{CacheManager, QueryEngine, QueryFilter};
use serde_json::{Value, json};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::Instant;
use test_helpers::synthetic_corpus as corpus;

// ==================== Budgets (ms) ====================
//
// Tighten here. MCP budget = in-process budget + `MCP_BUDGET_EXTRA_MS`.

/// One query shape from the field test.
struct Shape {
    name: &'static str,
    pattern: &'static str,
    /// `None` = fetch everything (in-process) / `mode: "count"` (MCP).
    limit: Option<usize>,
    regex: bool,
    /// In-process median budget, ms.
    budget_ms: f64,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "zero_hit",
        pattern: corpus::ABSENT,
        limit: Some(100),
        regex: false,
        budget_ms: 5.0,
    },
    Shape {
        name: "rare_ident",
        pattern: corpus::RARE_MARKER,
        limit: Some(100),
        regex: false,
        budget_ms: 20.0,
    },
    Shape {
        name: "common_ident_limit1",
        pattern: corpus::COMMON_IDENT,
        limit: Some(1),
        regex: false,
        budget_ms: 50.0,
    },
    Shape {
        name: "common_word_limit1",
        pattern: corpus::COMMON_WORD,
        limit: Some(1),
        regex: false,
        budget_ms: 50.0,
    },
    Shape {
        name: "common_word_count",
        pattern: corpus::COMMON_WORD,
        limit: None,
        regex: false,
        budget_ms: 150.0,
    },
    Shape {
        name: "regex_getset",
        pattern: corpus::GETSET_REGEX,
        limit: None,
        regex: true,
        budget_ms: 300.0,
    },
];

/// Added to every in-process budget for the stdio round-trip.
const MCP_BUDGET_EXTRA_MS: f64 = 15.0;

/// Repeats per shape.
const RUNS: usize = 11;

// ==================== Measurement ====================

/// A reported hit count. A list-mode search with a `limit` stops verifying once
/// the page is full, so its `total` is a lower bound and `upper` (candidate lines
/// from the index) an upper bound; count mode and no-limit searches are exact.
#[derive(Clone, Copy, Debug)]
struct Hits {
    total: usize,
    exact: bool,
    upper: Option<usize>,
}

impl Hits {
    fn exact(total: usize) -> Self {
        Self {
            total,
            exact: true,
            upper: None,
        }
    }

    /// The count to show in a table: `1234` or `216+`.
    fn label(&self) -> String {
        if self.exact {
            self.total.to_string()
        } else {
            format!("{}+", self.total)
        }
    }

    /// Whether `want` is consistent with this report.
    fn admits(&self, want: usize) -> bool {
        if self.exact {
            self.total == want
        } else {
            self.total <= want && self.upper.is_none_or(|u| u >= want)
        }
    }
}

struct Stats {
    hits: Hits,
    first_ms: f64,
    median_ms: f64,
    p90_ms: f64,
}

fn summarise(hits: Hits, samples_ms: &[f64]) -> Stats {
    let first_ms = samples_ms[0];
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let median_ms = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    // Nearest-rank p90.
    let p90_ms = sorted[((0.9 * n as f64).ceil() as usize).clamp(1, n) - 1];
    Stats {
        hits,
        first_ms,
        median_ms,
        p90_ms,
    }
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).map(|v| v == "1").unwrap_or(false)
}

fn print_table(harness: &str, rows: &[(&str, Stats)]) {
    println!("\n### {harness}\n");
    println!("| shape | hits | first | median | p90 |");
    println!("|---|---:|---:|---:|---:|");
    for (name, s) in rows {
        println!(
            "| {name} | {} | {:.2} | {:.2} | {:.2} |",
            s.hits.label(),
            s.first_ms,
            s.median_ms,
            s.p90_ms
        );
    }
    if env_flag("REFLEX_LATENCY_JSON") {
        for (name, s) in rows {
            println!(
                "{}",
                json!({
                    "harness": harness,
                    "shape": name,
                    "hits": s.hits.total,
                    "hits_exact": s.hits.exact,
                    "hits_upper": s.hits.upper,
                    "first_ms": s.first_ms,
                    "median_ms": s.median_ms,
                    "p90_ms": s.p90_ms,
                })
            );
        }
    }
}

/// Budget check: only enforced under `REFLEX_LATENCY_BUDGET=1`.
fn check_budgets(harness: &str, rows: &[(&str, Stats)], extra_ms: f64) {
    if !env_flag("REFLEX_LATENCY_BUDGET") {
        return;
    }
    let mut failures = Vec::new();
    for (shape, (name, s)) in SHAPES.iter().zip(rows) {
        let budget = shape.budget_ms + extra_ms;
        if s.median_ms > budget {
            failures.push(format!(
                "{harness}/{name}: median {:.2} ms > budget {budget:.0} ms",
                s.median_ms
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "latency budget exceeded:\n  {}",
        failures.join("\n  ")
    );
}

// ==================== Parity oracle ====================

/// Expected per-line hit counts from a plain scan of the generated files.
/// Literal shapes use whole-identifier (`\b…\b`) semantics, matching Reflex's
/// default; the regex shape uses the pattern as-is. One hit per matching line.
fn expected_hits(root: &Path) -> Vec<usize> {
    let mut files = Vec::new();
    collect_rs(&root.join("src"), &mut files);
    let contents: Vec<String> = files
        .iter()
        .map(|p| fs::read_to_string(p).expect("read corpus file"))
        .collect();

    SHAPES
        .iter()
        .map(|shape| {
            let re = if shape.regex {
                regex::Regex::new(shape.pattern).unwrap()
            } else {
                regex::Regex::new(&format!(r"\b{}\b", regex::escape(shape.pattern))).unwrap()
            };
            contents
                .iter()
                .map(|c| c.lines().filter(|l| re.is_match(l)).count())
                .sum()
        })
        .collect()
}

fn collect_rs(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).expect("read corpus dir") {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn count_files(root: &Path) -> usize {
    let mut files = Vec::new();
    collect_rs(&root.join("src"), &mut files);
    files.len()
}

/// Sanity asserts that always run.
fn assert_sanity(harness: &str, root: &Path, rows: &[(&str, Stats)], expected: &[usize]) {
    let files = count_files(root);
    for ((shape, (name, s)), &want) in SHAPES.iter().zip(rows).zip(expected) {
        assert!(
            s.hits.admits(want),
            "{harness}/{name}: Reflex reported {:?}, scan of generated files found {want}",
            s.hits
        );
        // Shapes without a limit (or with nothing to find) must be exact.
        if shape.limit.is_none() || want == 0 {
            assert!(
                s.hits.exact,
                "{harness}/{name}: {:?} should be exact",
                s.hits
            );
        }
        match shape.name {
            "zero_hit" => assert_eq!(s.hits.total, 0, "{harness}/{name}"),
            "rare_ident" => {
                assert_eq!(s.hits.total, corpus::RARE_MARKER_LINES, "{harness}/{name}")
            }
            "regex_getset" => assert!(
                s.hits.total >= files,
                "{harness}/{name}: {} hits < {files} files",
                s.hits.total
            ),
            _ => assert!(
                want > 1000,
                "{harness}/{name}: common shape hit only {want} lines"
            ),
        }
    }
}

// ==================== In-process harness ====================

fn filter_for(shape: &Shape) -> QueryFilter {
    QueryFilter {
        limit: shape.limit,
        use_regex: shape.regex,
        suppress_output: true,
        ..Default::default()
    }
}

/// Run one shape once; returns (hits, elapsed ms). Builds a fresh engine per
/// call to mirror the MCP handler.
fn run_in_process(root: &Path, shape: &Shape) -> (Hits, f64) {
    let start = Instant::now();
    let engine = QueryEngine::new(CacheManager::new(root));
    let response = engine
        .search_with_metadata(shape.pattern, filter_for(shape))
        .expect("query failed");
    let p = &response.pagination;
    let hits = Hits {
        total: p.total,
        exact: p.total_is_exact,
        upper: p.approx_total,
    };
    (hits, ms(start))
}

#[test]
#[ignore = "latency harness; run with --release"]
fn latency_in_process() {
    let root = corpus::indexed(corpus::DEFAULT_SEED);
    let expected = expected_hits(&root);

    let rows: Vec<(&str, Stats)> = SHAPES
        .iter()
        .map(|shape| {
            let mut hits = Hits::exact(0);
            let samples: Vec<f64> = (0..RUNS)
                .map(|_| {
                    let (h, t) = run_in_process(&root, shape);
                    hits = h;
                    t
                })
                .collect();
            (shape.name, summarise(hits, &samples))
        })
        .collect();

    print_table("in_process", &rows);
    assert_sanity("in_process", &root, &rows, &expected);
    check_budgets("in_process", &rows, 0.0);
}

// ==================== MCP stdio harness ====================

struct McpChild {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    next_id: u64,
}

impl McpChild {
    fn spawn(root: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_rfx"))
            .arg("mcp")
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn rfx mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            stdout,
            next_id: 1,
        }
    }

    fn send(&mut self, msg: &Value) {
        writeln!(self.stdin, "{msg}").expect("write to rfx mcp");
        self.stdin.flush().expect("flush rfx mcp stdin");
    }

    /// Send a request and block on its one-line response.
    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.stdout.read_line(&mut line).expect("read from rfx mcp");
            assert!(n > 0, "rfx mcp closed stdout while waiting for {method}");
            if !line.trim().is_empty() {
                break;
            }
        }
        let v: Value = serde_json::from_str(&line).expect("rfx mcp emitted invalid JSON");
        assert_eq!(v["id"], json!(id), "response id mismatch: {line}");
        assert!(v.get("error").is_none(), "rfx mcp error: {line}");
        v["result"].clone()
    }

    fn initialize(&mut self) -> f64 {
        let start = Instant::now();
        self.request(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "latency_budget", "version": "0"}
            }),
        );
        self.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        ms(start)
    }

    /// Call the tool for `shape`; returns (hits, elapsed ms).
    fn call_shape(&mut self, shape: &Shape) -> (Hits, f64) {
        let (tool, args) = mcp_call_for(shape);
        let start = Instant::now();
        let result = self.request("tools/call", json!({"name": tool, "arguments": args}));
        let elapsed = ms(start);

        assert_ne!(result["isError"], json!(true), "{}: {result}", shape.name);
        let text = result["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("{}: no content[0].text in {result}", shape.name));
        let body: Value = serde_json::from_str(text).expect("tool result text is JSON");
        let hits = match body["count"].as_u64() {
            Some(n) => Hits::exact(n as usize),
            None => Hits {
                total: body["pagination"]["total"]
                    .as_u64()
                    .or_else(|| body["total_count"].as_u64())
                    .unwrap_or_else(|| panic!("{}: no hit count in {body}", shape.name))
                    as usize,
                exact: body["total_is_exact"].as_bool().unwrap_or(true),
                upper: body["approx_total"].as_u64().map(|n| n as usize),
            },
        };
        (hits, elapsed)
    }
}

impl Drop for McpChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Map a shape onto the MCP tool an agent would use for it.
fn mcp_call_for(shape: &Shape) -> (&'static str, Value) {
    let tool = if shape.regex {
        "search_regex"
    } else {
        "search_code"
    };
    let args = match shape.limit {
        Some(limit) => json!({"pattern": shape.pattern, "limit": limit}),
        None => json!({"pattern": shape.pattern, "mode": "count"}),
    };
    (tool, args)
}

#[test]
#[ignore = "latency harness; run with --release"]
fn latency_mcp_stdio() {
    let root = corpus::indexed(corpus::DEFAULT_SEED);
    let expected = expected_hits(&root);

    let mut mcp = McpChild::spawn(&root);
    let initialize_ms = mcp.initialize();

    // The very first tool call pays the cold open (mmap, sqlite, freshness).
    let (_, first_call_ms) = mcp.call_shape(&SHAPES[0]);

    let rows: Vec<(&str, Stats)> = SHAPES
        .iter()
        .map(|shape| {
            let mut hits = Hits::exact(0);
            let samples: Vec<f64> = (0..RUNS)
                .map(|_| {
                    let (h, t) = mcp.call_shape(shape);
                    hits = h;
                    t
                })
                .collect();
            (shape.name, summarise(hits, &samples))
        })
        .collect();

    print_table("mcp", &rows);
    println!("\ninitialize_ms: {initialize_ms:.2}");
    println!(
        "first_call_ms: {first_call_ms:.2}  (shape: {}, includes cold open)",
        SHAPES[0].name
    );
    if env_flag("REFLEX_LATENCY_JSON") {
        println!(
            "{}",
            json!({"harness": "mcp", "initialize_ms": initialize_ms, "first_call_ms": first_call_ms})
        );
    }

    drop(mcp);
    assert_sanity("mcp", &root, &rows, &expected);
    check_budgets("mcp", &rows, MCP_BUDGET_EXTRA_MS);
}

// ==================== Always-on: corpus generator contract ====================

/// Small-scale check that the generator is deterministic and plants exactly
/// what the harness's sanity asserts rely on. Runs in the normal test suite.
#[test]
fn synthetic_corpus_is_deterministic_and_planted() {
    let a = tempfile::TempDir::new().unwrap();
    let b = tempfile::TempDir::new().unwrap();
    let (files, bytes) = (20, 200 * 1024);
    corpus::generate(a.path(), 42, files, bytes);
    corpus::generate(b.path(), 42, files, bytes);

    let mut paths_a = Vec::new();
    collect_rs(&a.path().join("src"), &mut paths_a);
    paths_a.sort();
    assert_eq!(paths_a.len(), files);

    let mut total = 0;
    for pa in &paths_a {
        let rel = pa.strip_prefix(a.path()).unwrap();
        let ca = fs::read_to_string(pa).unwrap();
        let cb = fs::read_to_string(b.path().join(rel)).unwrap();
        assert_eq!(ca, cb, "same seed must produce identical {}", rel.display());
        total += ca.len();
    }
    // Within 25% of the byte target.
    assert!(
        (total as f64) > 0.75 * bytes as f64 && (total as f64) < 1.25 * bytes as f64,
        "corpus is {total} bytes, target {bytes}"
    );

    let expected = expected_hits(a.path());
    let by_name: Vec<(&str, usize)> = SHAPES
        .iter()
        .map(|s| s.name)
        .zip(expected.iter().copied())
        .collect();
    let get = |n: &str| by_name.iter().find(|(k, _)| *k == n).unwrap().1;
    assert_eq!(get("zero_hit"), 0);
    assert_eq!(get("rare_ident"), corpus::RARE_MARKER_LINES);
    assert_eq!(get("regex_getset"), 2 * files);
    // Same pattern, so the count shape and the limit-1 shape share an oracle.
    assert_eq!(get("common_word_count"), get("common_word_limit1"));
    assert!(get("common_word_count") > 0);
    assert!(get("common_ident_limit1") > 0);

    // Marker short-circuits regeneration.
    assert!(a.path().join(".generated-42").exists());
}
