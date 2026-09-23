//! The CLI must fit in the smallest main-thread stack any target gives it.
//!
//! Windows allocates 1 MiB to the main thread; Linux and macOS give 8 MiB. In a debug
//! build, where nothing is inlined, clap's generated parser for Reflex's command tree
//! needs between 1 and 2 MiB — so on Windows EVERY `rfx` invocation aborted with
//! STATUS_STACK_OVERFLOW (`0xC00000FD` / exit -1073741571) and no message beyond
//! "thread 'main' has overflowed its stack".
//!
//! It went unnoticed because the Windows CI job had been failing at clippy since the
//! test that spawns the binary was added, so it never ran. `src/main.rs` now does the
//! work on an explicitly sized thread; these tests stop that regressing.

use clap::Parser;
use std::process::Command;

const RFX: &str = env!("CARGO_BIN_EXE_rfx");

/// Run `f` on a thread with a constrained stack.
///
/// Note what this can and cannot do: a Rust stack overflow is a fatal process abort,
/// not a catchable panic, so `join()` never reports one. If the budget is exceeded
/// the whole test binary dies with "has overflowed its stack" — still a loud CI
/// failure, just an abrupt one. The bool result therefore only distinguishes a normal
/// panic from success.
fn on_small_stack<F: FnOnce() + Send + 'static>(mib: usize, f: F) -> bool {
    std::thread::Builder::new()
        .stack_size(mib * 1024 * 1024)
        .spawn(f)
        .expect("spawn")
        .join()
        .is_ok()
}

/// The regression itself: parsing must not need more than Windows' main thread has.
///
/// This is the assertion that would have caught the bug. It fails on a 1 MiB stack
/// with the pre-fix arrangement, where `Cli::parse()` ran on `main`.
#[test]
fn argument_parsing_fits_in_a_windows_sized_stack() {
    // 2 MiB, not 1: measured, parsing overflows at 1 MiB and fits in 2, and `main.rs`
    // gives the worker 16 MiB. This pins the order of magnitude rather than an exact
    // frame layout, so it will not flap on a compiler upgrade — while still failing
    // loudly if parsing ever grows past a few MiB again.
    assert!(
        on_small_stack(2, || {
            let r = reflex::cli::Cli::try_parse_from(["rfx", "index", ".", "--quiet"]);
            assert!(r.is_ok(), "parse should succeed");
        }),
        "clap parsing overflowed a 2 MiB stack — main.rs must size the worker thread"
    );
}

/// Every subcommand's parser, not just `index`.
#[test]
fn every_subcommand_parses_within_that_stack() {
    let cases: Vec<Vec<&str>> = vec![
        vec!["rfx", "index", "."],
        vec!["rfx", "query", "pattern"],
        vec!["rfx", "query", "pattern", "--symbols", "--json"],
        vec!["rfx", "query", "--pattern", "-> Result<"],
        vec!["rfx", "query", "--", "-> Result<"],
        vec!["rfx", "deps", "src/main.rs"],
        vec!["rfx", "serve", "--port", "7878"],
        vec!["rfx", "mcp"],
        vec!["rfx", "watch"],
    ];
    for args in cases {
        let label = args.join(" ");
        assert!(
            on_small_stack(2, move || {
                let _ = reflex::cli::Cli::try_parse_from(args);
            }),
            "`{label}` overflowed a 2 MiB stack"
        );
    }
}

/// Moving parsing off `main` must not change what clap reports.
///
/// clap exits via `std::process::exit`, which ends the process from any thread — but
/// that is exactly the sort of thing worth pinning rather than assuming.
#[test]
fn clap_exit_codes_survive_running_off_the_main_thread() {
    for (args, want, what) in [
        (vec!["--version"], 0, "--version"),
        (vec!["--help"], 0, "--help"),
        (vec!["--definitely-not-a-flag"], 2, "unknown flag"),
        // `query` with no pattern exits 1, not clap's 2 — an empty pattern is
        // accepted by the parser and rejected later. Verified identical in the
        // released 1.7.0 binary, so it is the baseline, not a regression.
        (vec!["query"], 1, "missing pattern"),
    ] {
        let out = Command::new(RFX).args(&args).output().expect("spawn rfx");
        assert_eq!(
            out.status.code(),
            Some(want),
            "{what}: expected exit {want}, got {:?}\nstderr: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// `--version` and `--help` must still actually print.
#[test]
fn clap_output_survives_running_off_the_main_thread() {
    let v = Command::new(RFX).arg("--version").output().unwrap();
    let text = String::from_utf8_lossy(&v.stdout);
    assert!(text.contains("rfx"), "--version printed: {text:?}");

    let h = Command::new(RFX).arg("--help").output().unwrap();
    let text = String::from_utf8_lossy(&h.stdout);
    assert!(text.contains("index"), "--help should list subcommands");
    assert!(text.contains("query"), "--help should list subcommands");
}
