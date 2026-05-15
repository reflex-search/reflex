//! Reflex CLI entrypoint

use clap::Parser;

use reflex::cli::Cli;
use reflex::output;

fn main() {
    let cli = Cli::parse();

    if let Err(e) = cli.execute() {
        output::error(&format!("Error: {:#}", e));
        let code = e
            .downcast_ref::<reflex::errors::ReflexError>()
            .map(|re| re.exit_code())
            .unwrap_or(1);
        std::process::exit(code);
    }
}
