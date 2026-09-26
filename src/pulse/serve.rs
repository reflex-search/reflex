//! Serve a published Pulse site locally, under its base path.
//!
//! The site's base (`/` or `/reflex/`) comes from `.pulse-site.json`, so subpath
//! deployments can be checked before they ship.

use crate::pulse::publish::MARKER;
use anyhow::{Context, Result, bail};
use std::path::Path;

pub fn base_of(site: &Path) -> String {
    std::fs::read(site.join(MARKER))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v["base"].as_str().map(str::to_string))
        .unwrap_or_else(|| "/".into())
}

/// Serve `site` on `host:port` until interrupted.
pub fn serve(site: &Path, host: &str, port: u16, open: bool) -> Result<()> {
    if !site.join("index.html").exists() {
        bail!(
            "No built site at {}. Run `rfx pulse generate -o {}` first.",
            site.display(),
            site.display()
        );
    }
    let base = base_of(site);
    let prefix = base.trim_end_matches('/').to_string();
    let url = format!(
        "http://{host}:{port}{}",
        if prefix.is_empty() { "/" } else { &base }
    );
    let files = tower_http::services::ServeDir::new(site)
        .append_index_html_on_directories(true)
        .not_found_service(tower_http::services::ServeFile::new(site.join("404.html")));
    let app = if prefix.is_empty() {
        axum::Router::new().fallback_service(files)
    } else {
        let to = base.clone();
        axum::Router::new().nest_service(&prefix, files).route(
            "/",
            axum::routing::get(move || async move { axum::response::Redirect::temporary(&to) }),
        )
    };
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind((host, port))
            .await
            .with_context(|| format!("binding {host}:{port}"))?;
        eprintln!("Serving {} at {url} (Ctrl-C to stop)", site.display());
        if open {
            open_browser(&url);
        }
        axum::serve(listener, app).await.context("serving the site")
    })
}

fn open_browser(url: &str) {
    let r = if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else if cfg!(windows) {
        std::process::Command::new("cmd")
            .args(["/C", "start", url])
            .spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    if let Err(e) = r {
        eprintln!("Could not open a browser: {e}");
    }
}
