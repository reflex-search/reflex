//! Pulse: a documentation site generated from the Reflex index.
//!
//! ```text
//! index ─► extract (roles, API, surface, CLI) ─► build (Docs Model) ─► write (LLM slots)
//!       ─► render (HTML bundle) ─► runtime (Astro/Starlight) ─► publish (static HTML)
//! ```
//! Everything up to `render` is plain Rust and deterministic; the LLM pass only fills
//! narrative slots that already have structural fallbacks. See `site::generate_site`.

pub mod build;
pub mod changelog;
pub mod config;
pub mod dates;
pub mod diff;
pub mod extract;
pub mod glossary;
pub mod map;
pub mod model;
pub mod narrate;
pub mod publish;
pub mod render;
pub mod runtime;
pub mod serve;
pub mod site;
pub mod snapshot;
pub mod write;
