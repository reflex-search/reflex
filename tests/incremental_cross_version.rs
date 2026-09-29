//! A cache written by this binary, read and written by the last release (2.0.3),
//! which knows no manifest or delta. The release must never answer from the base
//! alone while a delta is live, and whatever it writes, this binary must report
//! the cache stale and rebuild it on its next index run.
//!
//! Needs the 2.0.3 binary at `/scratch/cache/rfx-pre-incremental` (built from the
//! commit before the incremental work); ignored by default:
//!   cargo test --release --test incremental_cross_version -- --ignored

use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

const OLD: &str = "/scratch/cache/rfx-pre-incremental";
const NEW: &str = env!("CARGO_BIN_EXE_rfx");

fn rfx(bin: &str, root: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(bin)
        .current_dir(root)
        .args(args)
        .output()
        .expect("run rfx");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr),
    )
}

/// `rfx query <token> --json`: (status, paths) or the error text.
fn query(bin: &str, root: &Path, token: &str) -> Result<(String, Vec<String>), String> {
    let (_, text) = rfx(bin, root, &["query", token, "--json"]);
    let json = text.lines().next().unwrap_or_default();
    let value: Value = serde_json::from_str(json).map_err(|_| text.clone())?;
    if let Some(error) = value.get("error") {
        return Err(error.to_string());
    }
    let paths = value["results"]
        .as_array()
        .map(|r| {
            r.iter()
                .map(|g| g["path"].as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default();
    Ok((
        value["status"].as_str().unwrap_or_default().to_string(),
        paths,
    ))
}

/// 100 files, so a 1-file edit stays under the delta's 5 % limit.
fn workspace() -> TempDir {
    let temp = TempDir::new().unwrap();
    let src = temp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    for i in 0..100 {
        fs::write(
            src.join(format!("f{i}.rs")),
            format!("fn filler_{i}() {{ common_token(); }}\n"),
        )
        .unwrap();
    }
    fs::write(src.join("b.rs"), "fn beta_old_token() {}\n").unwrap();
    temp
}

fn index(bin: &str, root: &Path) {
    let (ok, text) = rfx(bin, root, &["index", "--quiet"]);
    assert!(ok, "{bin} index: {text}");
}

fn old_binary() -> bool {
    if Path::new(OLD).exists() {
        true
    } else {
        eprintln!("skipped: no 2.0.3 binary at {OLD}");
        false
    }
}

#[test]
#[ignore]
fn the_release_reads_a_base_only_cache_as_stale() {
    if !old_binary() {
        return;
    }
    let temp = workspace();
    let root = temp.path();
    index(NEW, root);
    let (status, paths) = query(OLD, root, "beta_old_token").expect("2.0.3 reads the base");
    assert_eq!(status, "stale");
    assert_eq!(paths, ["src/b.rs"]);
}

#[test]
#[ignore]
fn the_release_stops_on_a_live_delta() {
    if !old_binary() {
        return;
    }
    let temp = workspace();
    let root = temp.path();
    index(NEW, root);
    fs::write(root.join("src/b.rs"), "fn beta_new_token() {}\n").unwrap();
    index(NEW, root);
    assert!(
        root.join(".reflex/delta.2.content.bin").exists(),
        "a live delta"
    );

    // The base still holds the old b.rs: 2.0.3 must not answer from it.
    let err = query(OLD, root, "beta_old_token").expect_err("2.0.3 must stop");
    assert!(err.contains("corrupted"), "{err}");
}

#[test]
#[ignore]
fn a_release_index_run_leaves_a_cache_this_binary_rebuilds() {
    if !old_binary() {
        return;
    }
    for live_delta in [false, true] {
        let temp = workspace();
        let root = temp.path();
        index(NEW, root);
        if live_delta {
            fs::write(root.join("src/f1.rs"), "fn filler_1_v2() {}\n").unwrap();
            index(NEW, root);
        }
        // The tree moves on and 2.0.3 indexes it (through the hard links, when
        // there are any).
        fs::write(root.join("src/b.rs"), "fn beta_new_token() {}\n").unwrap();
        index(OLD, root);

        // This binary never reports that cache fresh, and never fails on it.
        match query(NEW, root, "beta_new_token") {
            Ok((status, _)) => assert_eq!(status, "stale", "live delta: {live_delta}"),
            Err(e) => panic!("live delta: {live_delta}: {e}"),
        }
        // Its next index run rebuilds.
        index(NEW, root);
        assert_eq!(
            query(NEW, root, "beta_new_token").unwrap(),
            ("fresh".to_string(), vec!["src/b.rs".to_string()]),
            "live delta: {live_delta}"
        );
        assert_eq!(
            query(NEW, root, "beta_old_token").unwrap(),
            ("fresh".to_string(), vec![]),
            "live delta: {live_delta}"
        );
    }
}
