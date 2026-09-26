//! Public surface: which items a user of the code can reach, and under which path.
//!
//! Per language, a resolver walks from the package entry points (a crate's `lib.rs`,
//! a package's `exports`, …) and marks what is reachable. Everything else in a source
//! file is internal: it appears on Internals pages only.

pub mod rust;

pub use rust::{RustApi, RustCrate, RustItem, RustModule};
