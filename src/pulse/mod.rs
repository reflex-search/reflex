//! Pulse: Codebase intelligence surfaces
//!
//! Pulse turns Reflex from a tool you *query* into a tool that *tells you things*.
//! It takes structural facts the index already extracts — symbols, dependencies,
//! hotspots, file churn — and projects them into seven browsable surfaces:
//! a living wiki, a periodic digest, an architecture map, an onboarding guide,
//! a development timeline, a symbol glossary, and a visual explorer treemap.
//!
//! All surfaces are thin layers over the same core capability: **snapshot and diff**.

pub mod config;
pub mod snapshot;
pub mod diff;
pub mod digest;
pub mod wiki;
pub mod llm_cache;
pub mod narrate;
pub mod map;
pub mod site;
pub mod zola;
pub mod pagefind;
pub mod onboard;
pub mod git_intel;
pub mod glossary;
pub mod explorer;
