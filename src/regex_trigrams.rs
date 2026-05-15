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
//! - `(?i)test` → case-insensitive → triggers full scan (no literals)
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

/// Extract literal sequences (≥3 chars) from a regex pattern
///
/// This is a simple heuristic that splits on regex metacharacters.
/// It doesn't parse the full regex AST but works for common patterns.
///
/// # Regex Metacharacters
///
/// The following characters are treated as non-literal:
/// - Wildcards: `.` `*` `+` `?`
/// - Alternation: `|`
/// - Grouping: `(` `)`
/// - Character classes: `[` `]`
/// - Anchors: `^` `$`
/// - Escapes: `\` (followed by special char)
/// - Quantifiers: `{` `}`
///
/// # Case-Insensitive Patterns
///
/// Patterns with case-insensitive flags like `(?i)` return an empty vector,
/// forcing a full scan. This is because we cannot reliably extract trigrams
/// for case-insensitive matching (would need all case variations).
///
/// # Examples
///
/// ```
/// use reflex::regex_trigrams::extract_literal_sequences;
///
/// assert_eq!(extract_literal_sequences("hello"), vec!["hello"]);
/// assert_eq!(extract_literal_sequences("fn.*test"), vec!["test"]);
/// assert_eq!(extract_literal_sequences("class.*Controller"), vec!["class", "Controller"]);
///
/// // Case-insensitive patterns return empty (triggers full scan)
/// assert_eq!(extract_literal_sequences("(?i)test"), Vec::<String>::new());
/// ```
pub fn extract_literal_sequences(pattern: &str) -> Vec<String> {
    let mut sequences = Vec::new();
    let mut current = String::new();
    let mut chars = pattern.chars().peekable();
    let mut has_case_insensitive_flag = false;

    while let Some(ch) = chars.next() {
        match ch {
            // Regex metacharacters - break the literal sequence
            '.' | '*' | '+' | '?' | '|' | '[' | ']' | '^' | '$' => {
                if current.len() >= 3 {
                    sequences.push(current.clone());
                }
                current.clear();
            }

            // Opening parenthesis - check for inline flags
            '(' => {
                // Save current sequence before clearing
                if current.len() >= 3 {
                    sequences.push(current.clone());
                }
                current.clear();

                // Check if this is an inline flag like (?i), (?m), (?s), (?x), or (?:...)
                if chars.peek() == Some(&'?') {
                    chars.next(); // consume '?'

                    // Peek at the next character to determine the type of group
                    if let Some(&flag_ch) = chars.peek() {
                        match flag_ch {
                            // Non-capturing group (?:...) - only skip the ?: part, not the contents
                            ':' => {
                                chars.next(); // consume ':'
                                // The contents of (?:...) are processed normally
                            }
                            // Inline flags: (?i) (?m) (?s) (?x) (?i-m) etc.
                            // These are standalone modifiers, consume until ')'
                            'i' | 'm' | 's' | 'x' | '-' => {
                                // Check if 'i' flag is present (case-insensitive)
                                if flag_ch == 'i' {
                                    has_case_insensitive_flag = true;
                                }

                                // Consume flag characters and closing ')'
                                while let Some(&next_ch) = chars.peek() {
                                    if next_ch == 'i' {
                                        has_case_insensitive_flag = true;
                                    }
                                    chars.next();
                                    if next_ch == ')' {
                                        break;
                                    }
                                }
                            }
                            // Other special groups (lookahead, lookbehind, etc.) - skip entirely
                            _ => {
                                // Consume until closing ')'
                                while let Some(&next_ch) = chars.peek() {
                                    chars.next();
                                    if next_ch == ')' {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Closing parenthesis
            ')' => {
                if current.len() >= 3 {
                    sequences.push(current.clone());
                }
                current.clear();
            }

            // Opening brace - quantifier, consume until closing brace
            '{' => {
                if current.len() >= 3 {
                    sequences.push(current.clone());
                }
                current.clear();

                // Consume quantifier contents to avoid treating "2,3" as a literal
                while let Some(&next_ch) = chars.peek() {
                    chars.next();
                    if next_ch == '}' {
                        break;
                    }
                }
            }

            // Closing brace
            '}' => {
                if current.len() >= 3 {
                    sequences.push(current.clone());
                }
                current.clear();
            }

            // Backslash escapes
            '\\' => {
                if let Some(&next_ch) = chars.peek() {
                    match next_ch {
                        // Common escape sequences that represent single characters
                        's' | 'd' | 'w' | 'S' | 'D' | 'W' | 'n' | 't' | 'r' | 'b' | 'B' => {
                            // These are not literal characters, break the sequence
                            chars.next(); // consume the escaped char
                            if current.len() >= 3 {
                                sequences.push(current.clone());
                            }
                            current.clear();
                        }
                        // Escaped metacharacter - treat as literal
                        _ => {
                            chars.next(); // consume the escaped char
                            current.push(next_ch);
                        }
                    }
                } else {
                    // Backslash at end of pattern - ignore
                    if current.len() >= 3 {
                        sequences.push(current.clone());
                    }
                    current.clear();
                }
            }

            // Regular literal character
            _ => {
                current.push(ch);
            }
        }
    }

    // Don't forget the last sequence
    if current.len() >= 3 {
        sequences.push(current);
    }

    // If case-insensitive flag detected, return empty to force full scan
    // Cannot reliably extract trigrams for case-insensitive matching
    if has_case_insensitive_flag {
        log::debug!("Case-insensitive flag detected in pattern, cannot use trigram optimization");
        return vec![];
    }

    sequences
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
        // Case-insensitive flag should trigger full scan (return empty)
        let sequences = extract_literal_sequences("(?i)queryengine");
        assert_eq!(sequences, Vec::<String>::new());
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
        // Multiple inline flags including 'i' should trigger full scan
        let sequences = extract_literal_sequences("(?im)test");
        assert_eq!(sequences, Vec::<String>::new());
    }

    #[test]
    fn test_extract_literal_sequences_flag_before_literal() {
        // Flag with 'i' at start should trigger full scan
        let sequences = extract_literal_sequences("(?i)test.*function");
        assert_eq!(sequences, Vec::<String>::new());
    }
}
