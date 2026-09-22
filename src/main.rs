//! Reflex CLI entrypoint

use clap::Parser;

use reflex::cli::Cli;
use reflex::output;

/// Stack for the worker thread that actually runs the command.
///
/// Windows gives the main thread 1 MiB, against 8 MiB on Linux and macOS. Indexing a
/// large workspace exceeded that in a debug build and aborted with
/// STATUS_STACK_OVERFLOW (exit `0xC00000FD` / -1073741571) — `rfx index` dying with
/// no error message beyond "thread 'main' has overflowed its stack".
///
/// It is not unbounded recursion: the indexer itself completes on a 1 MiB stack on
/// Linux. The depth is in clap's generated parser — measured on Linux,
/// `Cli::try_parse_from` overflows at 1 MiB and needs between 1 and 2 MiB in a debug
/// build, where nothing is inlined. Reflex has a large command tree, so every `rfx`
/// invocation was affected on Windows, not just indexing; it only showed up in the
/// one test that spawns the binary rather than calling the library.
///
/// So ARGUMENT PARSING must run on the sized thread too. `clap` reports `--help`,
/// `--version` and usage errors by calling `std::process::exit`, which ends the
/// process from any thread, so its output and exit codes are unaffected.
const WORKER_STACK_BYTES: usize = 16 * 1024 * 1024;

fn main() {
    let worker = std::thread::Builder::new()
        .name("rfx".to_string())
        .stack_size(WORKER_STACK_BYTES)
        .spawn(|| {
            // Parsing lives here, not on main: see WORKER_STACK_BYTES.
            let cli = Cli::parse();
            if let Err(e) = cli.execute() {
                // Display error in red with clean formatting
                output::error(&format!("Error: {:#}", e));
                std::process::exit(1);
            }
        })
        .expect("failed to spawn worker thread");

    // A panic inside the worker has already printed its own message; exit non-zero
    // rather than reporting success.
    if worker.join().is_err() {
        std::process::exit(101);
    }
}
