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
/// It is not unbounded recursion: the same work completes on a 1 MiB stack on Linux,
/// so this is ordinary depth that Windows' smaller default cannot hold. Running the
/// command on an explicitly-sized thread makes the limit the same everywhere instead
/// of a platform accident.
const WORKER_STACK_BYTES: usize = 16 * 1024 * 1024;

fn main() {
    // Parse on the main thread so `--help`, `--version` and usage errors keep clap's
    // own exit codes and output.
    let cli = Cli::parse();

    let worker = std::thread::Builder::new()
        .name("rfx".to_string())
        .stack_size(WORKER_STACK_BYTES)
        .spawn(move || {
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
