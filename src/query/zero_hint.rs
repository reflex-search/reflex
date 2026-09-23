//! Why a search returned nothing, judged from the FILTER, in a fixed order.
//!
//! The 1.8.0 field test found the zero-result hint naming lock/generated files for
//! every kind of exclusion: `count_occurrences {pattern:"runs-on", file:".github/"}`
//! said "6 candidate file(s) were lock or generated files" when the true cause was
//! "hidden path, not indexed". An agent followed the hint, added `include_locks`,
//! got 0 again, and concluded the thing did not exist.
//!
//! One reason is chosen, the first that applies:
//!
//! 1. the filter names a hidden path (`.github/`, `.gitignore`) — not indexed;
//! 2. the `file` filter names a path that is not in the index — say why, from disk;
//! 3. every candidate under the filter was a lock/generated file — say how to widen;
//! 4. a whole-identifier search has substring hits — say how to see them;
//! 5. nothing applies — no hint. The generic "check spelling / broaden" text is
//!    enough, and an invented reason is worse than none.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::filter::{QueryFilter, excluded_by_default_hint_text, substring_hint_text};
use super::open_index::OpenIndex;
use crate::models::IndexConfig;

/// Machine-readable cause of a zero result, beside the prose `hint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExcludedReason {
    /// The filter names a dot-directory or dotfile, which the index skips
    /// (ripgrep's default) unless `[index] hidden = true`.
    Hidden,
    /// The `file` filter names a path that is not in the index: deleted, binary,
    /// ignored, over `max_file_size`, or added since the last index.
    NotIndexed,
    /// Every candidate under the filter was a lock or generated file.
    LockOrGenerated,
    /// Whole-identifier search: substring matches exist.
    WholeIdentifier,
}

/// A reason and the sentence that explains it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZeroHint {
    pub reason: ExcludedReason,
    pub text: String,
}

/// The first rule that applies to this zero result; `None` when none does.
#[allow(clippy::too_many_arguments)]
pub fn explain_zero(
    root: &Path,
    config: &IndexConfig,
    open: &OpenIndex,
    filter: &QueryFilter,
    pattern: &str,
    substring_only: Option<usize>,
    excluded_scoped: usize,
    bracket_rewritten: bool,
) -> Option<ZeroHint> {
    if !config.hidden
        && let Some(seg) = hidden_target(filter)
    {
        return Some(ZeroHint {
            reason: ExcludedReason::Hidden,
            text: format!(
                "Hidden paths ({seg}: dot-directories and dotfiles) are not indexed, \
                 matching ripgrep's default. Use grep --hidden for this path, or set \
                 [index] hidden = true in .reflex/config.toml and re-index."
            ),
        });
    }
    if let Some(text) = unindexed_target(root, config, open, filter) {
        return Some(ZeroHint {
            reason: ExcludedReason::NotIndexed,
            text,
        });
    }
    if excluded_scoped > 0 {
        return Some(ZeroHint {
            reason: ExcludedReason::LockOrGenerated,
            text: excluded_by_default_hint_text(excluded_scoped),
        });
    }
    match substring_only {
        Some(n) if n > 0 && !filter.use_contains && !bracket_rewritten => Some(ZeroHint {
            reason: ExcludedReason::WholeIdentifier,
            text: substring_hint_text(n, pattern),
        }),
        _ => None,
    }
}

/// The first hidden segment named by the `file` or `glob` filters, if any.
///
/// `.github/workflows`, `.gitignore`, `**/.githooks/**` all name one; `.`, `..`,
/// `*`, `**` and `*.yml` do not.
pub fn hidden_target(filter: &QueryFilter) -> Option<String> {
    filter
        .file_pattern
        .iter()
        .chain(filter.glob_patterns.iter())
        .flat_map(|p| p.split('/'))
        .find(|seg| crate::indexer::is_hidden_segment(seg))
        .map(str::to_string)
}

/// Rule 2: the `file` filter matches no indexed path. Says why, from the disk.
fn unindexed_target(
    root: &Path,
    config: &IndexConfig,
    open: &OpenIndex,
    filter: &QueryFilter,
) -> Option<String> {
    let fp = filter.file_pattern.as_deref()?;
    if fp.is_empty() {
        return None;
    }
    let needle = fp.strip_prefix("./").unwrap_or(fp);
    let any_indexed = (0..open.content.file_count() as u32).any(|id| {
        open.content
            .get_file_path(id)
            .and_then(|p| p.to_str())
            .is_some_and(|p| p.contains(needle))
    });
    if any_indexed {
        return None;
    }

    // The CLI's own heuristic: no wildcard, and a separator or an extension.
    let looks_like_path =
        !fp.contains('*') && !fp.contains('?') && (fp.contains('/') || fp.contains('.'));
    if !looks_like_path {
        return Some(format!("No indexed path contains {fp:?}."));
    }

    let full = root.join(needle.trim_end_matches('/'));
    let why = match std::fs::metadata(&full) {
        Err(_) => "not on disk — deleted since the last index".to_string(),
        Ok(md) if md.is_dir() => match crate::git::is_ignored(root, needle) {
            Some(true) => "a directory ignored by .gitignore".to_string(),
            _ => "a directory with no indexed file under it — run index_project if it \
                  was recently added"
                .to_string(),
        },
        Ok(md) if md.len() > config.max_file_size as u64 => {
            format!("larger than max_file_size ({} bytes)", config.max_file_size)
        }
        Ok(_) if crate::indexer::looks_binary(&full) => "binary".to_string(),
        Ok(_) => match crate::git::is_ignored(root, needle) {
            Some(true) => "ignored by .gitignore".to_string(),
            _ => "added since the last index — run index_project".to_string(),
        },
    };
    Some(format!("{fp} is not in the index ({why})."))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_file(fp: &str) -> QueryFilter {
        QueryFilter {
            file_pattern: Some(fp.to_string()),
            ..Default::default()
        }
    }

    fn with_glob(g: &str) -> QueryFilter {
        QueryFilter {
            glob_patterns: vec![g.to_string()],
            ..Default::default()
        }
    }

    #[test]
    fn hidden_segments_are_recognised() {
        assert_eq!(
            hidden_target(&with_file(".github/")).as_deref(),
            Some(".github")
        );
        assert_eq!(
            hidden_target(&with_file(".gitignore")).as_deref(),
            Some(".gitignore")
        );
        assert_eq!(
            hidden_target(&with_file("src/.env.example")).as_deref(),
            Some(".env.example")
        );
        assert_eq!(
            hidden_target(&with_glob("**/.githooks/**")).as_deref(),
            Some(".githooks")
        );
    }

    #[test]
    fn ordinary_segments_are_not_hidden() {
        for p in [
            "src/main.rs",
            "./src",
            "../lib",
            "*.yml",
            "**/*.rs",
            "Cargo.lock",
        ] {
            assert_eq!(hidden_target(&with_file(p)), None, "{p}");
            assert_eq!(hidden_target(&with_glob(p)), None, "{p}");
        }
        assert_eq!(hidden_target(&QueryFilter::default()), None);
    }
}
