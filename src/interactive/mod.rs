// Interactive mode modules
mod app;
mod effects;
mod filter_selector;
mod history;
mod input;
mod mouse;
mod results;
mod syntax;
mod terminal;
mod theme;
mod ui;

use anyhow::Result;
use app::InteractiveApp;

/// Main entry point for interactive mode
/// Launches the TUI and runs the event loop
/// Interactive search; searches update a stale index first unless `no_update`.
pub fn run_interactive(no_update: bool) -> Result<()> {
    let mut app = InteractiveApp::new(no_update)?;
    app.run()
}
