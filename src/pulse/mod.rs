//! Pulse: Codebase intelligence surfaces
//!
//! Pulse turns Reflex from a tool you *query* into a tool that *tells you things*.
//! It takes structural facts the index already extracts — symbols, dependencies,
//! hotspots, file churn — and projects them into three browsable surfaces:
//! a living wiki, a periodic digest, and an architecture map.
//!
//! All three surfaces are thin layers over the same core capability: **snapshot and diff**.

pub mod config;
pub mod snapshot;
pub mod diff;
pub mod digest;
pub mod wiki;
pub mod llm_cache;
pub mod map;
pub mod site;
