//! Bounded preview extraction, shared by every parser.
//!
//! # Why this module exists
//!
//! Until 1.7.2 each of the 14 active parsers carried its own identical copy of
//! `extract_preview`:
//!
//! ```ignore
//! let lines: Vec<&str> = source.lines().collect();
//! let end_idx = (start_idx + 7).min(lines.len());
//! lines[start_idx..end_idx].join("\n")
//! ```
//!
//! It is bounded in LINES and unbounded in BYTES. That is fine until a line stops
//! being a line. `vendor/swagger-ui-5.17.14/swagger-ui-bundle.js` is 1,452,753 bytes
//! on ONE line, so `lines.len()` is 1, `end_idx` is 1, and the function returns an
//! owned copy of the entire file — once per symbol.
//!
//! tree-sitter finds ~13,843 symbols in that bundle:
//!
//! ```text
//! 13,843 x 1,452,753 B  = 18.7 GiB
//! x2 (batch_set cloned) = 37.4 GiB
//! observed RSS          = 34.4 GiB
//! ```
//!
//! The `lines().collect()` ran per SYMBOL rather than per file, which is the other
//! half of the bug: a 3m55s run that should take seconds.
//!
//! # The fix
//!
//! Bound the preview in bytes as well as lines, and never materialise the line list.
//! [`extract_preview_at`] walks at most [`PREVIEW_MAX_BYTES`] bytes and stops, so cost
//! is O(bytes returned) plus the skip to the start line — not O(file) per symbol.

use crate::models::Span;

/// Maximum bytes in a parser-generated preview.
///
/// Every consumer truncates well below this already: the CLI at 100
/// (`cli::query::MAX_PREVIEW_LENGTH`), MCP at 180 (`DEFAULT_MCP_PREVIEW_LENGTH`) and
/// `semantic::answer` at 200. 512 leaves headroom for all of them while making the
/// pathological case impossible.
///
/// `--expand` is unaffected: it REPLACES the preview with the symbol body read from
/// content.bin (`query/mod.rs`), and has its own, larger ceiling.
pub const PREVIEW_MAX_BYTES: usize = 512;

/// How many lines a preview shows, when they fit in the byte budget.
const PREVIEW_MAX_LINES: usize = 7;

/// Largest index at or below `at` that is a UTF-8 character boundary.
///
/// `str::floor_char_boundary` is still unstable, and slicing a `String` at a raw byte
/// offset panics mid-codepoint. That is not hypothetical here — minified bundles
/// routinely carry emoji and CJK in embedded i18n tables.
fn floor_char_boundary(s: &str, at: usize) -> usize {
    if at >= s.len() {
        return s.len();
    }
    let mut i = at;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Truncate `s` to at most `max` bytes, never splitting a character.
pub fn truncate_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..floor_char_boundary(s, max)]
    }
}

/// Preview starting at a 0-indexed line: up to 7 lines or [`PREVIEW_MAX_BYTES`],
/// whichever ends first.
///
/// Returns `""` when `start_line_idx` is past the end of `source`, rather than
/// panicking as the old per-parser copies did.
pub fn extract_preview_at(source: &str, start_line_idx: usize) -> String {
    let mut out = String::new();

    // `skip().take()` rather than `lines().collect()`: for a one-line file the skip is
    // free, and we stop reading as soon as the byte budget is spent. The old code
    // allocated a Vec of every line in the file, for every symbol in the file.
    for line in source.lines().skip(start_line_idx).take(PREVIEW_MAX_LINES) {
        if !out.is_empty() {
            if out.len() + 1 > PREVIEW_MAX_BYTES {
                break;
            }
            out.push('\n');
        }

        let remaining = PREVIEW_MAX_BYTES - out.len();
        if line.len() <= remaining {
            out.push_str(line);
        } else {
            out.push_str(truncate_bytes(line, remaining));
            break;
        }
    }

    out
}

/// Preview for a span. The common case: 12 of the 14 parsers call this.
pub fn extract_preview(source: &str, span: &Span) -> String {
    extract_preview_at(source, span.start_line.saturating_sub(1))
}

/// Preview for a span inside an embedded script block (Vue, Svelte).
///
/// `line_offset` is where the `<script>` body starts in the outer file. Saturating
/// throughout: the old copies did `span.start_line - 1 - line_offset`, a usize
/// underflow that panics in debug and wraps in release.
pub fn extract_preview_offset(source: &str, span: &Span, line_offset: usize) -> String {
    extract_preview_at(
        source,
        span.start_line
            .saturating_sub(1)
            .saturating_sub(line_offset),
    )
}

/// Maximum bytes in a full-text or regex match preview.
///
/// The same units bug lives on the query path: `preview: line.to_string()` in
/// `query::QueryEngine` returns the whole matched LINE. On a minified bundle that is
/// 1.45 MB per hit, and at the default 200-result page size a single search would
/// build ~290 MB of previews before serialisation.
pub const LINE_PREVIEW_MAX_BYTES: usize = 512;

/// Maximum bytes in an `--expand` preview.
///
/// Expand's contract is "show more than a preview", so it gets its own, larger
/// ceiling. 32 KB is still 45x smaller than the swagger bundle, and a function body
/// past 32 KB has stopped being readable anyway.
pub const EXPAND_MAX_BYTES: usize = 32 * 1024;

/// A bounded preview of one matched line, centred on the match.
///
/// Head truncation is wrong here. A hit at byte 900,000 of a single-line bundle must
/// show its own neighbourhood; the first 512 bytes of the file tell the caller
/// nothing. `match_at` is the byte offset of the match within `line`.
///
/// Elision is marked with a leading and/or trailing `…` so a truncated preview is
/// never mistaken for the whole line.
pub fn line_preview(line: &str, match_at: usize) -> String {
    if line.len() <= LINE_PREVIEW_MAX_BYTES {
        return line.to_string();
    }

    // Leave room for the ellipsis markers.
    let budget = LINE_PREVIEW_MAX_BYTES.saturating_sub(2);
    let half = budget / 2;

    let raw_start = match_at.saturating_sub(half);
    let start = {
        let mut i = raw_start.min(line.len());
        while i < line.len() && !line.is_char_boundary(i) {
            i += 1;
        }
        i
    };
    let end = floor_char_boundary(line, (start + budget).min(line.len()));

    let mut out = String::with_capacity(LINE_PREVIEW_MAX_BYTES);
    if start > 0 {
        out.push('…');
    }
    out.push_str(&line[start..end]);
    if end < line.len() {
        out.push('…');
    }
    out
}

/// A bounded `--expand` body.
pub fn expand_preview(body: &str) -> String {
    if body.len() <= EXPAND_MAX_BYTES {
        return body.to_string();
    }
    let mut out = truncate_bytes(body, EXPAND_MAX_BYTES).to_string();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(line: usize) -> Span {
        Span {
            start_line: line,
            end_line: line,
        }
    }

    /// The 34.4 GiB bug, in one assertion.
    #[test]
    fn a_single_enormous_line_is_capped() {
        let source = "x".repeat(2_000_000);
        let preview = extract_preview(&source, &span(1));
        assert!(
            preview.len() <= PREVIEW_MAX_BYTES,
            "got {} bytes from a 2 MB single line",
            preview.len()
        );
    }

    /// The bug is multiplicative, so the fix must hold per symbol, not just in total.
    #[test]
    fn every_symbol_on_one_line_stays_bounded() {
        let source = "x".repeat(1_000_000);
        let total: usize = (0..1000)
            .map(|_| extract_preview(&source, &span(1)).len())
            .sum();
        assert!(
            total <= 1000 * PREVIEW_MAX_BYTES,
            "1000 previews totalled {total} bytes"
        );
    }

    #[test]
    fn normal_source_is_unchanged_and_shows_seven_lines() {
        let source = (1..=20)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let preview = extract_preview(&source, &span(3));
        assert_eq!(
            preview,
            "line 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9"
        );
    }

    #[test]
    fn a_short_file_returns_what_it_has() {
        assert_eq!(extract_preview("only one line", &span(1)), "only one line");
        assert_eq!(extract_preview("a\nb", &span(1)), "a\nb");
    }

    /// The old code did `lines[start_idx..end_idx]`, which panics here.
    #[test]
    fn a_start_line_past_the_end_returns_empty_instead_of_panicking() {
        assert_eq!(extract_preview("a\nb\nc", &span(99)), "");
        assert_eq!(extract_preview("", &span(1)), "");
        assert_eq!(extract_preview("a", &span(0)), "a", "0 must not underflow");
    }

    /// The old code did `span.start_line - 1 - line_offset`, a usize underflow.
    #[test]
    fn an_offset_larger_than_the_start_line_does_not_underflow() {
        let source = "a\nb\nc";
        assert_eq!(extract_preview_offset(source, &span(1), 10), "a\nb\nc");
    }

    /// Minified bundles carry emoji and CJK in embedded i18n tables. Slicing a byte
    /// offset mid-codepoint panics.
    #[test]
    fn truncation_lands_on_a_character_boundary() {
        for filler in ["日", "😀", "é"] {
            let source = filler.repeat(500_000);
            let preview = extract_preview(&source, &span(1));
            assert!(preview.len() <= PREVIEW_MAX_BYTES);
            assert!(!preview.is_empty(), "{filler} produced nothing");
            // Round-tripping proves the slice is well-formed UTF-8.
            assert_eq!(
                String::from_utf8(preview.clone().into_bytes()).unwrap(),
                preview
            );
        }
    }

    #[test]
    fn a_long_line_among_normal_ones_is_capped_but_still_returned() {
        // The `rbac_pb.ts` shape: one base64 descriptor among ordinary code.
        let source = format!("fn a() {{}}\n{}\nfn b() {{}}", "z".repeat(100_000));
        let preview = extract_preview(&source, &span(2));
        assert!(preview.len() <= PREVIEW_MAX_BYTES);
        assert!(preview.starts_with('z'));
    }

    #[test]
    fn truncate_bytes_never_splits_a_character() {
        assert_eq!(truncate_bytes("日本語", 4), "日");
        assert_eq!(truncate_bytes("abc", 10), "abc");
        assert_eq!(truncate_bytes("日", 1), "");
    }
}

#[cfg(test)]
mod line_preview_tests {
    use super::*;

    #[test]
    fn a_short_line_is_returned_whole() {
        assert_eq!(line_preview("fn main() {}", 3), "fn main() {}");
    }

    #[test]
    fn a_huge_line_is_capped() {
        let line = "x".repeat(1_452_753);
        let out = line_preview(&line, 0);
        assert!(out.len() <= LINE_PREVIEW_MAX_BYTES + 4, "{}", out.len());
    }

    /// The point of a window: a match deep inside the line must be visible.
    #[test]
    fn the_window_is_centred_on_the_match() {
        let line = format!("{}NEEDLE{}", "a".repeat(900_000), "b".repeat(500_000));
        let out = line_preview(&line, 900_000);
        assert!(out.contains("NEEDLE"), "match not in window: {out:.80}");
        assert!(out.starts_with('…'), "elision not marked: {out:.20}");
        assert!(out.ends_with('…'));
    }

    #[test]
    fn a_match_near_the_start_has_no_leading_ellipsis() {
        let line = format!("NEEDLE{}", "b".repeat(100_000));
        let out = line_preview(&line, 0);
        assert!(out.starts_with("NEEDLE"), "{out:.40}");
        assert!(out.ends_with('…'));
    }

    #[test]
    fn windowing_lands_on_character_boundaries() {
        let line = "日".repeat(200_000);
        let out = line_preview(&line, 300_000);
        assert!(out.len() <= LINE_PREVIEW_MAX_BYTES + 4);
        assert_eq!(String::from_utf8(out.clone().into_bytes()).unwrap(), out);
    }

    #[test]
    fn expand_gets_a_larger_ceiling_than_a_preview() {
        let body = "y".repeat(100_000);
        let out = expand_preview(&body);
        assert!(out.len() <= EXPAND_MAX_BYTES + 4);
        assert!(out.len() > LINE_PREVIEW_MAX_BYTES, "expand must show more");
        assert_eq!(expand_preview("fn a() {}"), "fn a() {}");
    }
}
