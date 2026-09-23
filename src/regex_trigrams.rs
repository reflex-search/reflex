//! Extract literal sequences from regex patterns for trigram optimization
//!
//! This module implements literal extraction from regular expressions to enable
//! fast regex search using the trigram index. The key insight is that many regex
//! patterns contain literal substrings that can narrow down candidate files.
//!
//! # Strategy
//!
//! 1. Extract all literal sequences from the regex pattern (≥3 chars)
//! 2. Generate trigrams from those literals
//! 3. Use trigrams to find files containing ANY literal (UNION approach)
//! 4. Verify actual matches with the regex engine
//!
//! # File Selection: UNION vs INTERSECTION
//!
//! For correctness, we use **UNION** (files with ANY literal):
//! - Alternation `(a|b)` needs files with a OR b → UNION is correct ✓
//! - Sequential `a.*b` needs files with a AND b → UNION includes extra files but still correct ✓
//!
//! Trade-off: UNION may scan 2-3x more files for sequential patterns, but ensures
//! we never miss matches. Performance impact is minimal (<5ms) due to memory-mapped I/O.
//!
//! # Examples
//!
//! - `fn\s+test_.*` → extracts "test_" → searches files containing "test_"
//! - `(class|function)` → extracts ["class", "function"] → searches files with class OR function
//! - `class.*Controller` → extracts ["class", "Controller"] → searches files with class OR Controller
//! - `(?i)test` → extracts "test" flagged case-insensitive → case-folded trigram lookup
//! - `.*` → no literals → fall back to full scan
//!
//! # References
//!
//! - Russ Cox - Regular Expression Matching with a Trigram Index
//!   https://swtch.com/~rsc/regexp/regexp4.html

use crate::trigram::{Trigram, extract_trigrams};

/// Extract guaranteed trigrams from a regex pattern
///
/// Returns trigrams that MUST appear in any string matching the pattern.
/// These trigrams are used to narrow down candidate files before running
/// the full regex match.
///
/// # Algorithm (MVP - Simple Literal Extraction)
///
/// 1. Split pattern on regex metacharacters: . * + ? | ( ) [ ] { } ^ $ \
/// 2. Keep literal sequences of 3+ characters
/// 3. Extract trigrams from each literal sequence
/// 4. Return all trigrams
///
/// # Examples
///
/// ```
/// use reflex::regex_trigrams::extract_trigrams_from_regex;
///
/// // Simple literal
/// let trigrams = extract_trigrams_from_regex("extract_symbols");
/// assert!(!trigrams.is_empty());
///
/// // Pattern with wildcard
/// let trigrams = extract_trigrams_from_regex("fn.*test");
/// assert!(!trigrams.is_empty()); // Has "fn " and "test"
///
/// // No literals
/// let trigrams = extract_trigrams_from_regex(".*");
/// assert!(trigrams.is_empty()); // Must fall back to full scan
/// ```
pub fn extract_trigrams_from_regex(pattern: &str) -> Vec<Trigram> {
    let literals = extract_literal_sequences(pattern);

    if literals.is_empty() {
        log::debug!(
            "No literals found in regex pattern '{}', will fall back to full scan",
            pattern
        );
        return vec![];
    }

    log::debug!(
        "Extracted {} literal sequences from regex: {:?}",
        literals.len(),
        literals
    );

    // Extract trigrams from all literal sequences
    let mut all_trigrams = Vec::new();
    for literal in literals {
        let trigrams = extract_trigrams(&literal);
        all_trigrams.extend(trigrams);
    }

    // Deduplicate trigrams
    all_trigrams.sort_unstable();
    all_trigrams.dedup();

    log::debug!(
        "Extracted {} unique trigrams from regex pattern",
        all_trigrams.len()
    );
    all_trigrams
}

/// A literal every match of the regex must contain verbatim, with the case
/// flag in force where it appeared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexLiteral {
    /// The literal text (≥3 bytes).
    pub text: String,
    /// `true` when the literal sits under an `i` flag: the index must match it
    /// case-insensitively (see `TrigramIndex::search_candidates_fold`).
    pub case_insensitive: bool,
}

/// One open group while scanning a pattern.
struct Frame {
    /// The `i` flag in force inside this group.
    ci: bool,
    /// Index into the output where this group's literals begin.
    start: usize,
    /// Index into the output where the current alternation branch begins.
    branch_start: usize,
    /// An earlier branch of this group emitted no literal, so the group's
    /// literals are not a superset of its matches.
    branch_without_literal: bool,
}

impl Frame {
    fn new(ci: bool, at: usize) -> Self {
        Self {
            ci,
            start: at,
            branch_start: at,
            branch_without_literal: false,
        }
    }
}

/// Extract literal sequences (≥3 chars) from a regex pattern
///
/// This is a simple heuristic that splits on regex metacharacters.
/// It doesn't parse the full regex AST but works for common patterns.
///
/// # Contract
///
/// Every line the pattern matches contains at least one returned literal
/// verbatim (case-insensitively for a literal under an `i` flag). The union
/// over literals of their candidate lines is therefore a superset of the
/// matching lines. To keep that promise the scanner:
///
/// - drops the atom before `?`, `*` and `{0,n}` (`foobar?` → `fooba`);
/// - drops every literal of a group when one of its alternation branches has
///   none (`(abc|de)f` → nothing: a `def` line contains no literal), or when
///   the whole group is optional (`(abc)?x`);
/// - treats a character class as one non-literal atom (`[abc]def` → `def`);
/// - breaks the sequence at any escape it does not recognise as a single
///   literal character (`\p{Lu}`, `\x41`, `\A`, …);
/// - returns nothing under the `x` (verbose) flag, where whitespace and `#`
///   are not literal.
///
/// # Case-Insensitive Patterns
///
/// A literal under `(?i)`, `(?i:...)` or `(?im)` is returned with
/// `case_insensitive: true`; `(?-i)` and the end of the enclosing group turn
/// the flag off again. Before 1.8.0 any `i` flag discarded every literal and
/// forced a full scan.
///
/// # Examples
///
/// ```
/// use reflex::regex_trigrams::{extract_literals, RegexLiteral};
///
/// let lits = extract_literals("(?i)test.*Func");
/// assert_eq!(lits.len(), 2);
/// assert!(lits.iter().all(|l| l.case_insensitive));
/// assert_eq!(lits[0].text, "test");
///
/// let lits = extract_literals("(?i:foo)bar");
/// assert_eq!(lits, vec![
///     RegexLiteral { text: "foo".into(), case_insensitive: true },
///     RegexLiteral { text: "bar".into(), case_insensitive: false },
/// ]);
/// ```
pub fn extract_literals(pattern: &str) -> Vec<RegexLiteral> {
    let mut out: Vec<RegexLiteral> = Vec::new();
    let mut current = String::new();
    let mut chars = pattern.chars().peekable();
    let mut stack: Vec<Frame> = vec![Frame::new(false, 0)];
    let mut verbose = false;
    // `Some(start)` right after a `)`: a following `?`, `*` or `{0,n}` makes the
    // whole group optional, so its literals are dropped rather than one char.
    let mut last_group_start: Option<usize> = None;
    // A `?` right after `*`, `+`, `?` or `}` is a laziness modifier, not a
    // quantifier: it must not pop another character.
    let mut prev_was_quantifier = false;

    fn flush(current: &mut String, out: &mut Vec<RegexLiteral>, ci: bool) {
        if current.len() >= 3 {
            out.push(RegexLiteral {
                text: std::mem::take(current),
                case_insensitive: ci,
            });
        } else {
            current.clear();
        }
    }

    while let Some(ch) = chars.next() {
        let ci = stack.last().map(|f| f.ci).unwrap_or(false);
        let mut group_just_closed = None;
        let mut is_quantifier = false;

        match ch {
            // A quantifier that allows ZERO repetitions binds only the atom before
            // it: in `foobar?` the `r` is optional, so the literal a match must
            // contain is `fooba`, not `foobar`. Emitting `foobar` would skip a file
            // (or, with line-level candidates, a line) that matches as `fooba`.
            '?' if prev_was_quantifier => {
                // Lazy modifier (`*?`, `+?`, `??`, `{n,m}?`): nothing new is optional.
                is_quantifier = true;
            }
            '*' | '?' => {
                is_quantifier = true;
                match last_group_start {
                    Some(start) => out.truncate(start),
                    None => {
                        current.pop();
                    }
                }
                flush(&mut current, &mut out, ci);
            }

            // Regex metacharacters - break the literal sequence. `+` needs at least
            // one repetition, so the atom before it stays part of the literal.
            '+' => {
                is_quantifier = true;
                flush(&mut current, &mut out, ci);
            }
            '.' | '^' | '$' => {
                flush(&mut current, &mut out, ci);
            }

            // Character class: one atom that matches a single character. Its body
            // is never literal (`[abc]def` must not yield `abc`).
            '[' => {
                flush(&mut current, &mut out, ci);
                let mut depth = 1usize;
                let mut first = true;
                while let Some(c) = chars.next() {
                    match c {
                        '\\' => {
                            chars.next();
                        }
                        '^' if first => continue,
                        ']' if first => {}
                        '[' => depth += 1,
                        ']' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    first = false;
                }
            }
            ']' => {
                flush(&mut current, &mut out, ci);
            }

            // Alternation: the branch that just ended must have emitted a literal,
            // or the enclosing group's literals cannot be a superset.
            '|' => {
                flush(&mut current, &mut out, ci);
                let frame = stack.last_mut().expect("root frame");
                if out.len() == frame.branch_start {
                    frame.branch_without_literal = true;
                }
                frame.branch_start = out.len();
            }

            // Opening parenthesis - a group, possibly with inline flags
            '(' => {
                flush(&mut current, &mut out, ci);

                if chars.peek() == Some(&'?') {
                    chars.next(); // consume '?'
                    match chars.peek().copied() {
                        // Non-capturing group (?:...): the body is scanned normally
                        Some(':') => {
                            chars.next();
                            stack.push(Frame::new(ci, out.len()));
                        }
                        // Named group (?P<name>...) / (?<name>...)
                        Some('P') | Some('<') => {
                            for c in chars.by_ref() {
                                if c == '>' {
                                    break;
                                }
                            }
                            stack.push(Frame::new(ci, out.len()));
                        }
                        // Inline flags: (?i) (?im) (?-i) (?i:...) (?x) ...
                        Some(c) if c.is_ascii_alphabetic() || c == '-' => {
                            let mut negate = false;
                            let mut new_ci = ci;
                            let mut scoped = false;
                            for c in chars.by_ref() {
                                match c {
                                    '-' => negate = true,
                                    'i' => new_ci = !negate,
                                    'x' => {
                                        if !negate {
                                            verbose = true;
                                        }
                                    }
                                    ':' => {
                                        scoped = true;
                                        break;
                                    }
                                    ')' => break,
                                    _ => {}
                                }
                            }
                            if scoped {
                                stack.push(Frame::new(new_ci, out.len()));
                            } else if let Some(frame) = stack.last_mut() {
                                // `(?i)` applies to the rest of the enclosing group
                                frame.ci = new_ci;
                            }
                        }
                        // Anything else (lookaround syntax the regex crate rejects
                        // anyway): skip to the closing parenthesis.
                        _ => {
                            for c in chars.by_ref() {
                                if c == ')' {
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    stack.push(Frame::new(ci, out.len()));
                }
            }

            // Closing parenthesis
            ')' => {
                flush(&mut current, &mut out, ci);
                if stack.len() > 1 {
                    let frame = stack.pop().expect("non-root frame");
                    let last_branch_has_literal = out.len() > frame.branch_start;
                    if frame.branch_without_literal || !last_branch_has_literal {
                        out.truncate(frame.start);
                    }
                    group_just_closed = Some(frame.start);
                }
            }

            // Opening brace - quantifier, consume until closing brace
            '{' => {
                is_quantifier = true;
                // `{0,n}` / `{0}` make the preceding atom optional, exactly like `?`;
                // any other lower bound keeps it required.
                let mut body = String::new();
                while let Some(&next_ch) = chars.peek() {
                    chars.next();
                    if next_ch == '}' {
                        break;
                    }
                    body.push(next_ch);
                }
                let min_repeats = body
                    .split(',')
                    .next()
                    .and_then(|n| n.trim().parse::<u32>().ok());
                if min_repeats == Some(0) {
                    match last_group_start {
                        Some(start) => out.truncate(start),
                        None => {
                            current.pop();
                        }
                    }
                }

                flush(&mut current, &mut out, ci);
            }

            // Closing brace
            '}' => {
                flush(&mut current, &mut out, ci);
            }

            // Backslash escapes
            '\\' => {
                match chars.next() {
                    // Escapes that stand for a class, an anchor, a boundary, a
                    // code point or a control character: never a literal byte
                    // in the sequence. `\x41`, `\u{..}`, `\p{..}` take a body.
                    Some(c) if c.is_ascii_alphanumeric() => {
                        flush(&mut current, &mut out, ci);
                        match c {
                            'x' | 'u' | 'U' | 'p' | 'P' => {
                                if chars.peek() == Some(&'{') {
                                    for c in chars.by_ref() {
                                        if c == '}' {
                                            break;
                                        }
                                    }
                                } else {
                                    let n = match c {
                                        'x' => 2,
                                        'u' => 4,
                                        'U' => 8,
                                        _ => 1,
                                    };
                                    for _ in 0..n {
                                        chars.next();
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    // Escaped metacharacter - treat as literal
                    Some(c) => current.push(c),
                    // Backslash at end of pattern - ignore
                    None => flush(&mut current, &mut out, ci),
                }
            }

            // Regular literal character
            _ => {
                current.push(ch);
            }
        }

        last_group_start = group_just_closed;
        prev_was_quantifier = is_quantifier;
    }

    // Don't forget the last sequence
    let ci = stack.last().map(|f| f.ci).unwrap_or(false);
    flush(&mut current, &mut out, ci);

    // The root is a group too: `abc|de` has a branch with no literal.
    let root = &stack[0];
    if root.branch_without_literal || out.len() == root.branch_start {
        out.clear();
    }

    if verbose {
        log::debug!("Verbose (x) flag in pattern, cannot use trigram optimization");
        return vec![];
    }

    out
}

/// Extract literal sequences (≥3 chars) from a regex pattern, ignoring case flags
///
/// The texts of [`extract_literals`]. A literal under `(?i)` is included; the
/// caller must match it case-insensitively.
///
/// # Examples
///
/// ```
/// use reflex::regex_trigrams::extract_literal_sequences;
///
/// assert_eq!(extract_literal_sequences("hello"), vec!["hello"]);
/// assert_eq!(extract_literal_sequences("fn.*test"), vec!["test"]);
/// assert_eq!(extract_literal_sequences("class.*Controller"), vec!["class", "Controller"]);
/// assert_eq!(extract_literal_sequences("(?i)test"), vec!["test"]);
/// ```
pub fn extract_literal_sequences(pattern: &str) -> Vec<String> {
    extract_literals(pattern)
        .into_iter()
        .map(|l| l.text)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_literal_sequences_simple() {
        let sequences = extract_literal_sequences("hello");
        assert_eq!(sequences, vec!["hello"]);
    }

    #[test]
    fn test_extract_literal_sequences_with_wildcard() {
        let sequences = extract_literal_sequences("fn.*test");
        assert_eq!(sequences, vec!["test"]);
    }

    #[test]
    fn test_extract_literal_sequences_multiple() {
        let sequences = extract_literal_sequences("class.*Controller");
        assert_eq!(sequences, vec!["class", "Controller"]);
    }

    #[test]
    fn test_extract_literal_sequences_no_literals() {
        let sequences = extract_literal_sequences(".*");
        assert!(sequences.is_empty());
    }

    #[test]
    fn test_extract_literal_sequences_short_literals() {
        // "fn" is only 2 chars, should be skipped
        let sequences = extract_literal_sequences("fn.*test");
        assert_eq!(sequences, vec!["test"]);
    }

    #[test]
    fn test_extract_literal_sequences_escaped() {
        // \. is escaped period, should be literal
        let sequences = extract_literal_sequences("test\\.txt");
        assert_eq!(sequences, vec!["test.txt"]);
    }

    #[test]
    fn test_extract_literal_sequences_whitespace_escape() {
        // \s is whitespace class, not literal
        let sequences = extract_literal_sequences("fn\\s+extract");
        assert_eq!(sequences, vec!["extract"]);
    }

    #[test]
    fn test_extract_literal_sequences_word_boundary() {
        // \b is word boundary, not literal
        let sequences = extract_literal_sequences("\\bListUsersController\\b");
        assert_eq!(sequences, vec!["ListUsersController"]);
    }

    #[test]
    fn test_extract_trigrams_simple_literal() {
        let trigrams = extract_trigrams_from_regex("extract");
        // "extract" has 5 trigrams: "ext", "xtr", "tra", "rac", "act"
        assert_eq!(trigrams.len(), 5);
    }

    #[test]
    fn test_extract_trigrams_with_wildcard() {
        let trigrams = extract_trigrams_from_regex("fn.*test");
        // "test" has 2 trigrams: "tes", "est"
        assert_eq!(trigrams.len(), 2);
    }

    #[test]
    fn test_extract_trigrams_multiple_literals() {
        let trigrams = extract_trigrams_from_regex("class.*Controller");
        // "class" has 3 trigrams, "Controller" has 8
        // Total unique: 11
        assert!(trigrams.len() >= 10); // At least 10 unique trigrams
    }

    #[test]
    fn test_extract_trigrams_no_literals() {
        let trigrams = extract_trigrams_from_regex(".*");
        assert!(trigrams.is_empty());
    }

    #[test]
    fn test_extract_trigrams_complex_pattern() {
        // "(function|const)\s+\w+\s*=" has "function" and "const" as literals
        let trigrams = extract_trigrams_from_regex("(function|const)");
        // "function" has 6 trigrams, "const" has 3
        assert!(trigrams.len() >= 6);
    }

    #[test]
    fn test_extract_literal_sequences_alternation() {
        // Alternation patterns should extract all alternatives as separate literals
        let sequences = extract_literal_sequences("(SymbolWriter|ContentWriter)");
        assert_eq!(sequences, vec!["SymbolWriter", "ContentWriter"]);
    }

    #[test]
    fn test_extract_literal_sequences_three_way_alternation() {
        // Three-way alternation
        let sequences = extract_literal_sequences("(Indexer|QueryEngine|CacheManager)");
        assert_eq!(sequences, vec!["Indexer", "QueryEngine", "CacheManager"]);
    }

    #[test]
    fn test_extract_literal_sequences_case_insensitive_flag() {
        // The literal survives; it is flagged for a case-folded lookup.
        assert_eq!(extract_literals("(?i)queryengine"), vec![ci("queryengine")]);
        assert_eq!(
            extract_literal_sequences("(?i)queryengine"),
            vec!["queryengine"]
        );
    }

    fn ci(text: &str) -> RegexLiteral {
        RegexLiteral {
            text: text.into(),
            case_insensitive: true,
        }
    }

    fn cs(text: &str) -> RegexLiteral {
        RegexLiteral {
            text: text.into(),
            case_insensitive: false,
        }
    }

    #[test]
    fn test_extract_literals_scoped_case_flag() {
        // `(?i:...)` covers only its body; the old scanner swallowed `foo`.
        assert_eq!(extract_literals("(?i:foo)bar"), vec![ci("foo"), cs("bar")]);
        // `(?-i)` turns the flag off for the rest of the group.
        assert_eq!(
            extract_literals("(?i)foo(?-i)bar"),
            vec![ci("foo"), cs("bar")]
        );
        // The flag ends with the enclosing group.
        assert_eq!(extract_literals("((?i)foo)bar"), vec![ci("foo"), cs("bar")]);
        assert_eq!(extract_literals("(?i-s:foo)"), vec![ci("foo")]);
        assert_eq!(extract_literals("(?s-i:foo)"), vec![cs("foo")]);
        // Non-ASCII is returned as-is; the engine decides it cannot fold it.
        assert_eq!(extract_literals("(?i)straße"), vec![ci("straße")]);
    }

    #[test]
    fn test_extract_literals_alternation_branch_without_literal() {
        // A `def` line contains no `abc`: the group's literals are not a superset.
        assert_eq!(extract_literal_sequences("(abc|de)f"), Vec::<String>::new());
        assert_eq!(extract_literal_sequences("(abc|de)fgh"), vec!["fgh"]);
        assert_eq!(extract_literal_sequences("abc|de"), Vec::<String>::new());
        assert_eq!(extract_literal_sequences("abc|"), Vec::<String>::new());
        assert_eq!(extract_literal_sequences("(abc|def)"), vec!["abc", "def"]);
        assert_eq!(
            extract_literal_sequences("(?i:abc)|de"),
            Vec::<String>::new()
        );
        // Nested: the inner group is fine, the outer branch is not.
        assert_eq!(extract_literal_sequences("((abc|def)|x)yyy"), vec!["yyy"]);
    }

    #[test]
    fn test_extract_literals_character_class_is_not_literal() {
        assert_eq!(extract_literal_sequences("[abc]def"), vec!["def"]);
        assert_eq!(extract_literal_sequences("[^abc]def"), vec!["def"]);
        assert_eq!(extract_literal_sequences("[]abc]def"), vec!["def"]);
        assert_eq!(extract_literal_sequences("[[:alpha:]]def"), vec!["def"]);
        assert_eq!(extract_literal_sequences(r"[\]abc]def"), vec!["def"]);
        assert_eq!(
            extract_literal_sequences("abc[xyz]?def"),
            vec!["abc", "def"]
        );
    }

    #[test]
    fn test_extract_literals_unknown_escapes_break_sequence() {
        assert_eq!(extract_literal_sequences(r"abc\p{Lu}"), vec!["abc"]);
        assert_eq!(extract_literal_sequences(r"abc\pLdef"), vec!["abc", "def"]);
        assert_eq!(extract_literal_sequences(r"abc\x41def"), vec!["abc", "def"]);
        assert_eq!(
            extract_literal_sequences(r"abc\x{41}def"),
            vec!["abc", "def"]
        );
        assert_eq!(
            extract_literal_sequences(r"abc\u{1F600}def"),
            vec!["abc", "def"]
        );
        assert_eq!(extract_literal_sequences(r"\Aabc\z"), vec!["abc"]);
        assert_eq!(extract_literal_sequences(r"abc\-def"), vec!["abc-def"]);
        assert_eq!(extract_literal_sequences(r"abc\/def"), vec!["abc/def"]);
    }

    #[test]
    fn test_extract_literals_verbose_flag_forces_scan() {
        assert_eq!(
            extract_literal_sequences("(?x) abc # comment"),
            Vec::<String>::new()
        );
        assert_eq!(extract_literal_sequences("(?ix:abc)"), Vec::<String>::new());
    }

    #[test]
    fn test_extract_literals_named_group() {
        assert_eq!(
            extract_literal_sequences("(?P<name>abc)def"),
            vec!["abc", "def"]
        );
        assert_eq!(extract_literal_sequences("(?<name>abc)?def"), vec!["def"]);
    }

    #[test]
    fn test_extract_literal_sequences_multiline_flag() {
        // Multiline flag should be skipped
        let sequences = extract_literal_sequences("(?m)^test");
        assert_eq!(sequences, vec!["test"]);
    }

    #[test]
    fn test_extract_literal_sequences_non_capturing_group() {
        // Non-capturing group (?:...) should not extract flag chars
        let sequences = extract_literal_sequences("(?:test|func)");
        assert_eq!(sequences, vec!["test", "func"]);
    }

    /// `?` and `*` make the atom before them optional: the required literal ends
    /// one character earlier. `+` keeps it. Line-level regex candidates rely on
    /// every emitted literal being present verbatim in every match.
    #[test]
    fn test_extract_literal_sequences_optional_last_atom() {
        assert_eq!(extract_literal_sequences("foobar?"), vec!["fooba"]);
        assert_eq!(extract_literal_sequences("abcd*"), vec!["abc"]);
        assert_eq!(extract_literal_sequences("abcd+"), vec!["abcd"]);
        assert_eq!(extract_literal_sequences("ab{0,2}c"), Vec::<String>::new());
        assert_eq!(
            extract_literal_sequences("test{0}word"),
            vec!["tes", "word"]
        );
        assert_eq!(
            extract_literal_sequences("test{1,5}word"),
            vec!["test", "word"]
        );
        // The optional atom is a group: every literal of the group is dropped.
        assert_eq!(extract_literal_sequences("foobar(baz)?"), vec!["foobar"]);
        assert_eq!(extract_literal_sequences("foobar(baz)*"), vec!["foobar"]);
        assert_eq!(
            extract_literal_sequences("foobar(baz){0,3}"),
            vec!["foobar"]
        );
        assert_eq!(
            extract_literal_sequences("foobar(baz)+"),
            vec!["foobar", "baz"]
        );
        // A lazy modifier after a quantifier is not a second quantifier.
        assert_eq!(extract_literal_sequences("abcd+?"), vec!["abcd"]);
        assert_eq!(extract_literal_sequences("abcd*?x"), vec!["abc"]);
    }

    #[test]
    fn test_extract_literal_sequences_quantifier_no_false_literal() {
        // Quantifier contents should NOT become a literal
        let sequences = extract_literal_sequences("a{2,3}test");
        assert_eq!(sequences, vec!["test"]);

        // Ensure "2,3" is NOT extracted
        assert!(!sequences.contains(&"2,3".to_string()));
    }

    #[test]
    fn test_extract_literal_sequences_quantifier_range() {
        // Test various quantifier formats
        let sequences = extract_literal_sequences("test{1,5}word");
        assert_eq!(sequences, vec!["test", "word"]);
        assert!(!sequences.contains(&"1,5".to_string()));
    }

    #[test]
    fn test_extract_literal_sequences_quantifier_exact() {
        // Exact quantifier {n}
        let sequences = extract_literal_sequences("test{3}word");
        assert_eq!(sequences, vec!["test", "word"]);
        assert!(!sequences.contains(&"3".to_string()));
    }

    #[test]
    fn test_extract_literal_sequences_combined_flags() {
        // Multiple inline flags including 'i': literal kept, flagged
        assert_eq!(extract_literals("(?im)test"), vec![ci("test")]);
        assert_eq!(extract_literals("(?mi)^test"), vec![ci("test")]);
    }

    #[test]
    fn test_extract_literal_sequences_flag_before_literal() {
        // Flag with 'i' at start covers every literal after it
        assert_eq!(
            extract_literals("(?i)test.*function"),
            vec![ci("test"), ci("function")]
        );
    }
}
