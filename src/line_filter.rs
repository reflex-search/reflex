//! Line-based pre-filtering to detect comments and string literals
//!
//! This module provides language-specific filters that analyze lines of code
//! to determine if a pattern match occurs inside a comment or string literal.
//! This enables us to skip files where ALL matches are in non-code contexts,
//! avoiding expensive tree-sitter parsing when possible.
//!
//! # Performance Impact
//!
//! Pre-filtering can reduce tree-sitter parsing workload by 2-5x:
//! - Pattern "mb_" in Linux kernel: 2,500 files → ~500 files to parse
//! - Expected speedup: 38s → ~1-2s for symbol queries
//!
//! # Design Philosophy
//!
//! - **Conservative**: Only skip files when 100% certain ALL matches are in comments/strings
//! - **Language-specific**: Each language has its own comment/string syntax rules
//! - **Line-based**: Fast heuristic analysis without full parsing
//! - **No false negatives**: Never skip files with valid code matches
//!
//! # How a line is read
//!
//! One left-to-right scan per question ([`region_at`]). It knows each language's
//! comments, strings, raw strings, char literals and string interpolation, so the
//! `//` of `"https://…"` is part of a string, not the start of a comment, and
//! `${call()}` inside a template literal is code. Through 2.2.0 each filter found the
//! first `//` or `#` on the line wherever it was, and dropped every match after it.
//!
//! A line is read without the lines around it, so two cases count as code (the match
//! is kept): a string that does not close on this line (it may be the end of a string
//! that began earlier), and the inside of a block comment that began on an earlier line.

use crate::models::Language;

/// Trait for language-specific line filtering
pub trait LineFilter {
    /// Check if a position in a line is inside a comment
    ///
    /// # Arguments
    /// * `line` - The full line of text
    /// * `pattern_pos` - Byte position where the pattern starts (0-indexed)
    ///
    /// # Returns
    /// `true` if the pattern is definitely inside a comment, `false` otherwise
    fn is_in_comment(&self, line: &str, pattern_pos: usize) -> bool;

    /// Check if a position in a line is inside a string literal
    ///
    /// # Arguments
    /// * `line` - The full line of text
    /// * `pattern_pos` - Byte position where the pattern starts (0-indexed)
    ///
    /// # Returns
    /// `true` if the pattern is definitely inside a string literal, `false` otherwise
    fn is_in_string(&self, line: &str, pattern_pos: usize) -> bool;

    /// Check if a position in a line is inside a comment or a string literal, with
    /// one scan of the line instead of two.
    fn is_in_comment_or_string(&self, line: &str, pattern_pos: usize) -> bool {
        self.is_in_comment(line, pattern_pos) || self.is_in_string(line, pattern_pos)
    }
}

/// Get a LineFilter for a specific language
pub fn get_filter(lang: Language) -> Option<Box<dyn LineFilter>> {
    let syntax = match lang {
        Language::Rust => Syntax::Rust,
        Language::C | Language::Cpp => Syntax::C,
        Language::Go => Syntax::Go,
        Language::Java => Syntax::Java,
        // Vue and Svelte: JS/TS in <script> sections; templates read the same way.
        Language::JavaScript | Language::TypeScript | Language::Vue | Language::Svelte => {
            Syntax::JavaScript
        }
        Language::Python => Syntax::Python,
        Language::Ruby => Syntax::Ruby,
        Language::PHP => Syntax::Php,
        Language::CSharp => Syntax::CSharp,
        Language::Kotlin => Syntax::Kotlin,
        Language::Zig => Syntax::Zig,
        // No filter for the text tier: "inside a string literal" is not a meaningful
        // idea in markdown or YAML, and guessing would drop real matches.
        Language::Text
        | Language::Lock
        | Language::Generated
        | Language::Swift
        | Language::Unknown => return None,
    };
    Some(Box::new(SyntaxFilter(syntax)))
}

/// The comment and string rules of one language family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Syntax {
    Rust,
    /// C and C++ (C++ raw strings `R"x(…)x"` and digit separators `1'000` included).
    C,
    Go,
    Java,
    /// JavaScript, TypeScript, Vue and Svelte.
    JavaScript,
    Python,
    Ruby,
    Php,
    CSharp,
    Kotlin,
    Zig,
}

impl Syntax {
    fn slash_comments(self) -> bool {
        !matches!(self, Syntax::Python | Syntax::Ruby)
    }

    fn hash_comments(self) -> bool {
        matches!(self, Syntax::Python | Syntax::Ruby | Syntax::Php)
    }

    fn block_comments(self) -> bool {
        !matches!(self, Syntax::Python | Syntax::Ruby | Syntax::Zig)
    }

    fn nested_block_comments(self) -> bool {
        matches!(self, Syntax::Rust | Syntax::Kotlin)
    }

    /// `'x'` is a char literal; elsewhere `'` delimits a string.
    fn char_literals(self) -> bool {
        matches!(
            self,
            Syntax::Rust
                | Syntax::C
                | Syntax::Go
                | Syntax::Java
                | Syntax::CSharp
                | Syntax::Kotlin
                | Syntax::Zig
        )
    }
}

struct SyntaxFilter(Syntax);

impl LineFilter for SyntaxFilter {
    fn is_in_comment(&self, line: &str, pattern_pos: usize) -> bool {
        region_at(self.0, line, pattern_pos) == Region::Comment
    }

    fn is_in_string(&self, line: &str, pattern_pos: usize) -> bool {
        region_at(self.0, line, pattern_pos) == Region::Str
    }

    fn is_in_comment_or_string(&self, line: &str, pattern_pos: usize) -> bool {
        region_at(self.0, line, pattern_pos) != Region::Code
    }
}

/// What a position of a line is, as far as the line alone can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Region {
    /// Code, or not known (the conservative answer).
    Code,
    Comment,
    Str,
}

#[derive(Debug, Clone, Copy)]
enum Frame {
    Str(StrKind),
    /// Code inside a string (`${…}`, `#{…}`, `{…}`), with its open `{` count.
    Interp(u32),
}

#[derive(Debug, Clone, Copy)]
struct StrKind {
    close: Close,
    /// `\` escapes the next byte.
    backslash: bool,
    /// `""` is an escaped quote (C# verbatim strings).
    doubled_quote: bool,
    interp: Interp,
}

#[derive(Debug, Clone, Copy)]
enum Close {
    /// One delimiter byte: `"`, `'` or a backtick.
    Byte(u8),
    /// Three delimiter bytes: `"""` or `'''`.
    Triple(u8),
    /// Rust raw string: `"` and this many `#`.
    RustRaw(usize),
    /// C++ raw string: `)`, the delimiter at `line[start..end]`, `"`.
    CppRaw(usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Interp {
    None,
    /// `${…}` (JavaScript template literals).
    DollarBrace,
    /// `${…}` and `$name` (Kotlin).
    Kotlin,
    /// `{…}`, with `{{` and `}}` as literal braces (Python f-strings, C# `$"…"`).
    Brace,
    /// `#{…}` (Ruby).
    HashBrace,
    /// `{$…}` and `$name->prop` (PHP).
    Php,
}

impl StrKind {
    fn plain(quote: u8) -> Self {
        StrKind {
            close: Close::Byte(quote),
            backslash: true,
            doubled_quote: false,
            interp: Interp::None,
        }
    }

    fn raw(close: Close) -> Self {
        StrKind {
            close,
            backslash: false,
            doubled_quote: false,
            interp: Interp::None,
        }
    }

    fn with_interp(self, interp: Interp) -> Self {
        StrKind { interp, ..self }
    }
}

enum CodeStep {
    /// A line comment starts here: the rest of the line is a comment.
    LineComment,
    /// `[i, end)` is one token of this region (block comment, char literal, regex).
    Span(usize, Region),
    /// A string opens with an opener of this many bytes.
    Open(usize, StrKind),
    Byte,
}

enum StrStep {
    Skip(usize),
    /// The string closes with a delimiter of this many bytes.
    Close(usize),
    /// An interpolation opens with an opener of this many bytes.
    Interp(usize),
    /// `[i, end)` is code inside the string (`$name`).
    CodeSpan(usize),
}

/// Classify byte `pos` of `line`.
fn region_at(syntax: Syntax, line: &str, pos: usize) -> Region {
    let b = line.as_bytes();
    let mut stack: Vec<Frame> = Vec::new();
    // Set once the scan reaches `pos` inside a string: that string's depth on `stack`.
    // The answer is `Str` only if that string closes on this line.
    let mut open_at: Option<usize> = None;
    let mut i = 0;

    loop {
        if open_at.is_none() && i >= pos {
            match stack.last() {
                Some(Frame::Str(_)) => open_at = Some(stack.len()),
                _ => return Region::Code,
            }
        }
        if i >= b.len() {
            // The string that holds `pos` does not close on this line.
            return Region::Code;
        }
        let reached = open_at.is_some();

        match stack.last().copied() {
            Some(Frame::Str(kind)) => match string_step(b, i, kind) {
                StrStep::Skip(n) => i += n,
                StrStep::Close(n) => {
                    stack.pop();
                    i += n;
                    if open_at.is_some_and(|depth| stack.len() < depth) {
                        return Region::Str;
                    }
                }
                StrStep::Interp(n) => {
                    stack.push(Frame::Interp(0));
                    i += n;
                }
                StrStep::CodeSpan(end) => {
                    if !reached && pos < end {
                        return Region::Code;
                    }
                    i = end;
                }
            },
            top => match code_step(syntax, b, i) {
                CodeStep::LineComment => {
                    // Not reached: `pos` lies after the comment start.
                    return if reached {
                        Region::Code
                    } else {
                        Region::Comment
                    };
                }
                CodeStep::Span(end, region) => {
                    if !reached && pos < end {
                        return region;
                    }
                    i = end;
                }
                CodeStep::Open(n, kind) => {
                    if !reached && pos < i + n {
                        return Region::Code;
                    }
                    stack.push(Frame::Str(kind));
                    i += n;
                }
                CodeStep::Byte => {
                    if let Some(Frame::Interp(depth)) = top {
                        match b[i] {
                            b'{' => *stack.last_mut().unwrap() = Frame::Interp(depth + 1),
                            b'}' if depth == 0 => {
                                stack.pop();
                            }
                            b'}' => *stack.last_mut().unwrap() = Frame::Interp(depth - 1),
                            _ => {}
                        }
                    }
                    i += 1;
                }
            },
        }
    }
}

/// One step of the scan in code at `b[i]`.
fn code_step(syntax: Syntax, b: &[u8], i: usize) -> CodeStep {
    let next = b.get(i + 1).copied();
    match b[i] {
        b'/' if next == Some(b'/') && syntax.slash_comments() => CodeStep::LineComment,
        b'/' if next == Some(b'*') && syntax.block_comments() => {
            CodeStep::Span(block_comment_end(syntax, b, i), Region::Comment)
        }
        b'/' if syntax == Syntax::JavaScript && regex_can_start(b, i) => match regex_end(b, i) {
            Some(end) => CodeStep::Span(end, Region::Str),
            None => CodeStep::Byte,
        },
        // PHP 8 attributes (`#[Route(…)]`) are code.
        b'#' if syntax.hash_comments() && !(syntax == Syntax::Php && next == Some(b'[')) => {
            CodeStep::LineComment
        }
        // Zig multiline string literal: the rest of the line is the string.
        b'\\' if syntax == Syntax::Zig && next == Some(b'\\') => {
            CodeStep::Span(b.len(), Region::Str)
        }
        b'\'' if syntax.char_literals() => match char_literal_end(b, i) {
            Some(end) => CodeStep::Span(end, Region::Str),
            None => CodeStep::Byte,
        },
        b'"' | b'\'' | b'`' => match open_string(syntax, b, i) {
            Some((n, kind)) => CodeStep::Open(n, kind),
            None => CodeStep::Byte,
        },
        _ => CodeStep::Byte,
    }
}

/// The string that a quote at `b[i]` opens, and the length of its opener. A prefix
/// before the quote (`r#`, `f`, `@$`, `R`) was already read as code; it is looked
/// up behind the quote.
fn open_string(syntax: Syntax, b: &[u8], i: usize) -> Option<(usize, StrKind)> {
    let q = b[i];
    let triple = b.get(i..i + 3) == Some(&[q, q, q][..]);
    match syntax {
        Syntax::Rust => (q == b'"').then(|| (1, rust_string(b, i))),
        Syntax::C => (q == b'"').then(|| cpp_string(b, i)),
        Syntax::Go => match q {
            b'"' => Some((1, StrKind::plain(q))),
            b'`' => Some((1, StrKind::raw(Close::Byte(q)))),
            _ => None,
        },
        Syntax::Java | Syntax::Zig => match q {
            b'"' if triple && syntax == Syntax::Java => Some((
                3,
                StrKind {
                    close: Close::Triple(q),
                    ..StrKind::plain(q)
                },
            )),
            b'"' => Some((1, StrKind::plain(q))),
            _ => None,
        },
        Syntax::JavaScript => match q {
            b'`' => Some((1, StrKind::plain(q).with_interp(Interp::DollarBrace))),
            _ => Some((1, StrKind::plain(q))),
        },
        Syntax::Python => {
            if q == b'`' {
                return None;
            }
            // Prefixes: r, b, f, u, t and two-letter mixes (rb, fr, …).
            let prefix = ident_run_before(b, i);
            let f_string = prefix.len() <= 2
                && prefix.iter().all(|c| b"rRbBfFuUtT".contains(c))
                && prefix.iter().any(|c| b"fFtT".contains(c));
            let kind = if triple {
                StrKind {
                    close: Close::Triple(q),
                    ..StrKind::plain(q)
                }
            } else {
                StrKind::plain(q)
            };
            let kind = if f_string {
                kind.with_interp(Interp::Brace)
            } else {
                kind
            };
            Some((if triple { 3 } else { 1 }, kind))
        }
        Syntax::Ruby => match q {
            b'\'' => Some((1, StrKind::plain(q))),
            _ => Some((1, StrKind::plain(q).with_interp(Interp::HashBrace))),
        },
        Syntax::Php => match q {
            b'\'' => Some((1, StrKind::plain(q))),
            _ => Some((1, StrKind::plain(q).with_interp(Interp::Php))),
        },
        Syntax::CSharp => {
            if q != b'"' {
                return None;
            }
            let mut j = i;
            while j > 0 && matches!(b[j - 1], b'@' | b'$') {
                j -= 1;
            }
            let prefix = &b[j..i];
            let interp = if prefix.contains(&b'$') {
                Interp::Brace
            } else {
                Interp::None
            };
            if prefix.contains(&b'@') {
                // Verbatim: `\` is a plain byte, `""` is a quote.
                return Some((
                    1,
                    StrKind {
                        close: Close::Byte(q),
                        backslash: false,
                        doubled_quote: true,
                        interp,
                    },
                ));
            }
            if triple {
                // C# 11 raw string literal.
                return Some((3, StrKind::raw(Close::Triple(q)).with_interp(interp)));
            }
            Some((1, StrKind::plain(q).with_interp(interp)))
        }
        Syntax::Kotlin => match q {
            b'"' if triple => Some((
                3,
                StrKind::raw(Close::Triple(q)).with_interp(Interp::Kotlin),
            )),
            b'"' => Some((1, StrKind::plain(q).with_interp(Interp::Kotlin))),
            _ => None,
        },
    }
}

/// `"…"`, `b"…"`, or a raw string `r"…"`, `r#"…"#`, `br##"…"##`.
fn rust_string(b: &[u8], i: usize) -> StrKind {
    let mut j = i;
    while j > 0 && b[j - 1] == b'#' {
        j -= 1;
    }
    let hashes = i - j;
    let prefix = ident_run_before(b, j);
    if matches!(prefix, b"r" | b"br" | b"cr") {
        StrKind::raw(Close::RustRaw(hashes))
    } else {
        StrKind::plain(b'"')
    }
}

/// `"…"`, or a C++ raw string `R"delim(…)delim"` (also `LR`, `uR`, `UR`, `u8R`).
fn cpp_string(b: &[u8], i: usize) -> (usize, StrKind) {
    let prefix = ident_run_before(b, i);
    if matches!(prefix, b"R" | b"LR" | b"uR" | b"UR" | b"u8R") {
        let start = i + 1;
        let delim_len = b[start..]
            .iter()
            .take(17)
            .position(|&c| c == b'(')
            .filter(|&n| {
                b[start..start + n]
                    .iter()
                    .all(|c| !c.is_ascii_whitespace() && !b")\\\"".contains(c))
            });
        if let Some(n) = delim_len {
            let end = start + n;
            return (end + 1 - i, StrKind::raw(Close::CppRaw(start, end)));
        }
    }
    (1, StrKind::plain(b'"'))
}

/// One step of the scan inside a string at `b[i]`.
fn string_step(b: &[u8], i: usize, kind: StrKind) -> StrStep {
    let c = b[i];
    let next = b.get(i + 1).copied();
    if kind.backslash && c == b'\\' {
        return StrStep::Skip(2);
    }
    if kind.doubled_quote && c == b'"' && next == Some(b'"') {
        return StrStep::Skip(2);
    }
    if let Some(n) = closes(kind.close, b, i) {
        return StrStep::Close(n);
    }
    match (kind.interp, c, next) {
        (Interp::DollarBrace | Interp::Kotlin, b'$', Some(b'{')) => StrStep::Interp(2),
        (Interp::Kotlin, b'$', Some(n)) if is_ident_start(n) => {
            StrStep::CodeSpan(ident_end(b, i + 1))
        }
        (Interp::Brace, b'{', Some(b'{')) | (Interp::Brace, b'}', Some(b'}')) => StrStep::Skip(2),
        (Interp::Brace, b'{', _) => StrStep::Interp(1),
        (Interp::HashBrace, b'#', Some(b'{')) => StrStep::Interp(2),
        (Interp::Php, b'{', Some(b'$')) => StrStep::Interp(1),
        (Interp::Php, b'$', Some(n)) if is_ident_start(n) => {
            let mut end = ident_end(b, i + 1);
            while b[end..].starts_with(b"->") && b.get(end + 2).is_some_and(|&c| is_ident_start(c))
            {
                end = ident_end(b, end + 2);
            }
            StrStep::CodeSpan(end)
        }
        _ => StrStep::Skip(1),
    }
}

/// The length of the closing delimiter at `b[i]`, if one is there.
fn closes(close: Close, b: &[u8], i: usize) -> Option<usize> {
    let rest = &b[i..];
    match close {
        Close::Byte(q) => (rest[0] == q).then_some(1),
        Close::Triple(q) => rest.starts_with(&[q, q, q]).then_some(3),
        Close::RustRaw(hashes) => {
            (rest[0] == b'"' && rest.len() > hashes && rest[1..=hashes].iter().all(|&c| c == b'#'))
                .then_some(1 + hashes)
        }
        Close::CppRaw(start, end) => {
            let delim = &b[start..end];
            (rest[0] == b')'
                && rest[1..].starts_with(delim)
                && rest.get(1 + delim.len()) == Some(&b'"'))
            .then_some(2 + delim.len())
        }
    }
}

/// The end of the block comment that starts at `b[i]` (`/*`), or the line's end.
fn block_comment_end(syntax: Syntax, b: &[u8], i: usize) -> usize {
    let nested = syntax.nested_block_comments();
    let mut depth = 1;
    let mut j = i + 2;
    while j < b.len() {
        if b[j..].starts_with(b"*/") {
            depth -= 1;
            j += 2;
            if depth == 0 {
                return j;
            }
        } else if nested && b[j..].starts_with(b"/*") {
            depth += 1;
            j += 2;
        } else {
            j += 1;
        }
    }
    b.len()
}

/// The end of a char literal that starts at `b[i]` (`'`): one character, or one escape
/// (`'\n'`, `'\''`, `'\u{1F600}'`). `None` for a Rust lifetime or label (`'a`), a C++
/// digit separator (`1'000`) or anything longer, which stay code.
fn char_literal_end(b: &[u8], i: usize) -> Option<usize> {
    let body = i + 1;
    match *b.get(body)? {
        b'\\' => {
            let from = body + 2;
            let close = b.get(from..)?.iter().take(10).position(|&c| c == b'\'')?;
            Some(from + close + 1)
        }
        b'\'' => None,
        c => {
            let end = body + utf8_len(c);
            (b.get(end) == Some(&b'\'')).then_some(end + 1)
        }
    }
}

/// Whether a `/` at `b[i]` starts a JavaScript regex literal rather than a division:
/// the usual rule, by the token before it.
fn regex_can_start(b: &[u8], i: usize) -> bool {
    let mut j = i;
    while j > 0 && b[j - 1].is_ascii_whitespace() {
        j -= 1;
    }
    if j == 0 {
        return true;
    }
    let c = b[j - 1];
    if is_ident(c) || c == b'$' {
        return matches!(
            ident_run_before(b, j),
            b"return"
                | b"typeof"
                | b"instanceof"
                | b"in"
                | b"of"
                | b"new"
                | b"delete"
                | b"void"
                | b"throw"
                | b"case"
                | b"do"
                | b"else"
                | b"yield"
                | b"await"
        );
    }
    b"(,=:[!&|?{};+-*%<>~^".contains(&c)
}

/// The end of the regex literal that starts at `b[i]`, if it closes on this line.
fn regex_end(b: &[u8], i: usize) -> Option<usize> {
    let mut in_class = false;
    let mut j = i + 1;
    while j < b.len() {
        match b[j] {
            b'\\' => j += 1,
            b'[' => in_class = true,
            b']' => in_class = false,
            b'/' if !in_class => return Some(j + 1),
            _ => {}
        }
        j += 1;
    }
    None
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c >= 0x80
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_' || c >= 0x80
}

/// The identifier bytes that end just before `b[i]`.
fn ident_run_before(b: &[u8], i: usize) -> &[u8] {
    let mut j = i;
    while j > 0 && is_ident(b[j - 1]) {
        j -= 1;
    }
    &b[j..i]
}

fn ident_end(b: &[u8], start: usize) -> usize {
    let mut j = start;
    while j < b.len() && is_ident(b[j]) {
        j += 1;
    }
    j
}

/// The byte length of the UTF-8 character that starts with `c`.
fn utf8_len(c: u8) -> usize {
    match c {
        0xF0.. => 4,
        0xE0.. => 3,
        0xC0.. => 2,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TARGET: &str = "call_the_target";

    fn filter(lang: Language) -> Box<dyn LineFilter> {
        get_filter(lang).expect("a filter")
    }

    /// The region of the first `call_the_target` on `line`.
    fn region(lang: Language, line: &str) -> Region {
        let pos = line.find(TARGET).expect("the target is on the line");
        let f = filter(lang);
        match (f.is_in_comment(line, pos), f.is_in_string(line, pos)) {
            (true, false) => Region::Comment,
            (false, true) => Region::Str,
            (false, false) => Region::Code,
            (true, true) => panic!("both a comment and a string: {line}"),
        }
    }

    fn assert_regions(lang: Language, cases: &[(&str, Region)]) {
        for (line, want) in cases {
            assert_eq!(region(lang, line), *want, "{lang:?}: {line}");
            let pos = line.find(TARGET).unwrap();
            assert_eq!(
                filter(lang).is_in_comment_or_string(line, pos),
                *want != Region::Code,
                "{lang:?} one-scan answer: {line}"
            );
        }
    }

    use Region::{Code, Comment, Str};

    // ========================================================================
    // Rust Tests
    // ========================================================================

    #[test]
    fn test_rust_single_line_comment() {
        let filter = filter(Language::Rust);
        let line = "let x = 5; // extract_symbols here";
        assert!(filter.is_in_comment(line, 15)); // "extract" is in comment
        assert!(!filter.is_in_comment(line, 4)); // "x" is not in comment
    }

    #[test]
    fn test_rust_multiline_comment() {
        let filter = filter(Language::Rust);
        let line = "let x = /* extract_symbols */ 5;";
        assert!(filter.is_in_comment(line, 11)); // "extract" is in comment
        assert!(!filter.is_in_comment(line, 30)); // "5" is not in comment
    }

    #[test]
    fn test_rust_string_literal() {
        let filter = filter(Language::Rust);
        let line = r#"let s = "extract_symbols";"#;
        assert!(filter.is_in_string(line, 9)); // "extract" is in string
        assert!(!filter.is_in_string(line, 27)); // after string
    }

    #[test]
    fn test_rust_raw_string() {
        let filter = filter(Language::Rust);
        let line = r#"let s = r"extract_symbols";"#;
        assert!(filter.is_in_string(line, 10)); // "extract" is in raw string
    }

    #[test]
    fn test_rust_raw_string_with_hashes() {
        let filter = filter(Language::Rust);
        let line = r###"let s = r#"extract_symbols"#;"###;
        assert!(filter.is_in_string(line, 11)); // "extract" is in raw string
    }

    #[test]
    fn test_rust_escaped_quote() {
        let filter = filter(Language::Rust);
        let line = r#"let s = "before \" extract_symbols after";"#;
        assert!(filter.is_in_string(line, 15)); // "extract" is in string
    }

    #[test]
    fn rust_comment_markers_inside_strings_are_not_comments() {
        assert_regions(
            Language::Rust,
            &[
                (
                    r#"let url = "https://example.com/api"; call_the_target();"#,
                    Code,
                ),
                (r#"let g = "src/*.rs"; call_the_target();"#, Code),
                (r#"let s = "//"; // call_the_target"#, Comment),
                (r##"let s = r#"a "// b"#; call_the_target();"##, Code),
                (r#"let s = "a \"//\" b"; call_the_target();"#, Code),
            ],
        );
    }

    #[test]
    fn rust_char_literals_and_lifetimes() {
        assert_regions(
            Language::Rust,
            &[
                (r#"let q = '"'; call_the_target("x");"#, Code),
                (r#"let q = '\''; call_the_target("x");"#, Code),
                (r#"let q = b'"'; call_the_target("x");"#, Code),
                (r#"fn f<'a>(x: &'a str) { call_the_target("x") }"#, Code),
                (r#"'outer: loop { call_the_target("x") }"#, Code),
                ("let r#type = 1; call_the_target();", Code),
            ],
        );
    }

    #[test]
    fn rust_nested_block_comments() {
        assert_regions(
            Language::Rust,
            &[
                ("/* a /* b */ call_the_target */ x", Comment),
                ("/* a /* b */ c */ call_the_target()", Code),
            ],
        );
    }

    #[test]
    fn a_string_that_does_not_close_on_the_line_is_not_known() {
        // The end of a string that began on an earlier line, then code.
        assert_regions(Language::Rust, &[(r#"    ", call_the_target);"#, Code)]);
        // The start of a string that ends on a later line: also kept.
        assert_regions(Language::Rust, &[(r#"let s = "call_the_target"#, Code)]);
        // A block comment that does not close on the line is still a comment.
        assert_regions(Language::Rust, &[("x(); /* call_the_target", Comment)]);
    }

    // ========================================================================
    // C / C++ Tests
    // ========================================================================

    #[test]
    fn c_strings_char_literals_and_raw_strings() {
        assert_regions(
            Language::C,
            &[
                (r#"printf("/* not a comment */"); call_the_target();"#, Code),
                (r#"puts("http://x"); call_the_target();"#, Code),
                (r#"char c = '"'; call_the_target();"#, Code),
                ("int x = 1; // call_the_target", Comment),
                ("int x = /* call_the_target */ 1;", Comment),
            ],
        );
        assert_regions(
            Language::Cpp,
            &[
                (r#"auto s = R"x(call_the_target)x";"#, Str),
                (r#"auto s = R"x(a )" // b)x"; call_the_target();"#, Code),
                (r#"auto s = u8R"(//)"; call_the_target();"#, Code),
                ("int n = 1'000'000; call_the_target('a');", Code),
                ("int n = 1'000; call_the_target('a');", Code),
            ],
        );
    }

    // ========================================================================
    // Go Tests
    // ========================================================================

    #[test]
    fn test_go_raw_string() {
        let filter = filter(Language::Go);
        let line = "s := `extract_symbols`";
        assert!(filter.is_in_string(line, 6));
    }

    #[test]
    fn go_comment_markers_inside_strings_are_not_comments() {
        assert_regions(
            Language::Go,
            &[
                ("s := `//` + call_the_target()", Code),
                (r#"u := "https://x"; call_the_target()"#, Code),
                (r#"r := '"'; call_the_target()"#, Code),
                ("x := 1 // call_the_target", Comment),
            ],
        );
    }

    // ========================================================================
    // Java Tests
    // ========================================================================

    #[test]
    fn java_strings_and_comments() {
        assert_regions(
            Language::Java,
            &[
                (r#"String u = "https://x"; call_the_target();"#, Code),
                (r#"String s = "call_the_target";"#, Str),
                (r#"char c = '"'; call_the_target();"#, Code),
                ("/** call_the_target */", Comment),
            ],
        );
    }

    // ========================================================================
    // JavaScript Tests
    // ========================================================================

    #[test]
    fn test_js_single_line_comment() {
        let filter = filter(Language::JavaScript);
        let line = "let x = 5; // extract_symbols here";
        assert!(filter.is_in_comment(line, 15));
        assert!(!filter.is_in_comment(line, 4));
    }

    #[test]
    fn test_js_string_double_quote() {
        let filter = filter(Language::JavaScript);
        let line = r#"let s = "extract_symbols";"#;
        assert!(filter.is_in_string(line, 9));
        assert!(!filter.is_in_string(line, 27));
    }

    #[test]
    fn test_js_string_single_quote() {
        let filter = filter(Language::JavaScript);
        let line = "let s = 'extract_symbols';";
        assert!(filter.is_in_string(line, 9));
    }

    #[test]
    fn test_js_template_literal() {
        let filter = filter(Language::JavaScript);
        let line = "let s = `extract_symbols`;";
        assert!(filter.is_in_string(line, 9));
    }

    #[test]
    fn js_comment_markers_inside_strings_are_not_comments() {
        assert_regions(
            Language::JavaScript,
            &[
                (r#"const u = "https://x"; call_the_target(u);"#, Code),
                ("const u = 'http://x', v = call_the_target();", Code),
                // Minified: one early URL used to hide every later match.
                (
                    r#"var a="https://cdn.x/y.js",b=1;function f(){return call_the_target(a)}"#,
                    Code,
                ),
            ],
        );
    }

    #[test]
    fn js_template_interpolation_is_code() {
        assert_regions(
            Language::JavaScript,
            &[
                ("const s = `${call_the_target(user)}`;", Code),
                ("const s = `a ${x} call_the_target`;", Str),
                ("const s = `a ${ {k: 1}.k } call_the_target`;", Str),
                ("const s = `${`${call_the_target()}`}`;", Code),
                ("const s = `${'}'} call_the_target`;", Str),
            ],
        );
    }

    #[test]
    fn js_regex_literals() {
        assert_regions(
            Language::JavaScript,
            &[
                (r"const re = /^https?:\/\//; call_the_target(re);", Code),
                (r#"if (/"/.test(s)) call_the_target("x");"#, Code),
                (r"s.replace(/[/]/g, '-'); call_the_target();", Code),
                ("const n = a / b; // call_the_target", Comment),
                ("const n = a / b / call_the_target;", Code),
                ("return /call_the_target/.test(s);", Str),
            ],
        );
    }

    #[test]
    fn typescript_vue_and_svelte_read_like_javascript() {
        for lang in [Language::TypeScript, Language::Vue, Language::Svelte] {
            assert_regions(
                lang,
                &[
                    (r#"const u = "https://x"; call_the_target(u);"#, Code),
                    ("// call_the_target", Comment),
                ],
            );
        }
    }

    // ========================================================================
    // Python Tests
    // ========================================================================

    #[test]
    fn test_python_comment() {
        let filter = filter(Language::Python);
        let line = "x = 5  # extract_symbols here";
        assert!(filter.is_in_comment(line, 9));
        assert!(!filter.is_in_comment(line, 0));
    }

    #[test]
    fn test_python_string() {
        let filter = filter(Language::Python);
        let line = r#"s = "extract_symbols""#;
        assert!(filter.is_in_string(line, 5));
    }

    #[test]
    fn test_python_triple_quote() {
        let filter = filter(Language::Python);
        let line = r#"s = """extract_symbols""""#;
        assert!(filter.is_in_string(line, 7));
    }

    #[test]
    fn python_hash_inside_strings_is_not_a_comment() {
        assert_regions(
            Language::Python,
            &[
                (r##"color = "#fff"; call_the_target(color)"##, Code),
                ("tag = '#x'; call_the_target()", Code),
                (r##"s = """# a""" + call_the_target()"##, Code),
                ("x = 1  # call_the_target", Comment),
            ],
        );
    }

    #[test]
    fn python_f_string_interpolation_is_code() {
        assert_regions(
            Language::Python,
            &[
                (r#"s = f"{call_the_target()}""#, Code),
                (r#"s = rf"{call_the_target()}""#, Code),
                (r#"s = f"call_the_target {x}""#, Str),
                (r#"s = f"{{call_the_target}}""#, Str),
                (r#"s = r"{call_the_target}""#, Str),
            ],
        );
    }

    // ========================================================================
    // Ruby Tests
    // ========================================================================

    #[test]
    fn ruby_strings_comments_and_interpolation() {
        assert_regions(
            Language::Ruby,
            &[
                (r##"color = "#fff"; call_the_target"##, Code),
                (r##"s = "#{call_the_target}""##, Code),
                (r##"s = '#{call_the_target}'"##, Str),
                ("x = 1 # call_the_target", Comment),
            ],
        );
    }

    // ========================================================================
    // PHP Tests
    // ========================================================================

    #[test]
    fn php_strings_comments_attributes_and_interpolation() {
        assert_regions(
            Language::PHP,
            &[
                (r#"$u = "https://x"; call_the_target();"#, Code),
                (r##"$c = "#fff"; call_the_target();"##, Code),
                ("#[Route('/x')] function call_the_target() {}", Code),
                ("# call_the_target", Comment),
                ("// call_the_target", Comment),
                (r#"$s = "{$o->call_the_target()}";"#, Code),
                (r#"$s = "$call_the_target";"#, Code),
                (r#"$s = "$o->call_the_target";"#, Code),
                (r#"$s = '$call_the_target';"#, Str),
            ],
        );
    }

    // ========================================================================
    // C# Tests
    // ========================================================================

    #[test]
    fn test_csharp_verbatim_string() {
        let filter = filter(Language::CSharp);
        let line = r#"string s = @"extract_symbols";"#;
        assert!(filter.is_in_string(line, 13));
    }

    #[test]
    fn test_csharp_verbatim_escaped_quote() {
        let filter = filter(Language::CSharp);
        let line = r#"string s = @"before "" extract_symbols after";"#;
        assert!(filter.is_in_string(line, 19));
    }

    #[test]
    fn csharp_verbatim_and_interpolated_strings() {
        assert_regions(
            Language::CSharp,
            &[
                (r#"var p = @"C:\dir\"; call_the_target();"#, Code),
                (r#"var u = "https://x"; call_the_target();"#, Code),
                (r#"var s = $"{call_the_target()}";"#, Code),
                (r#"var s = $@"{call_the_target()}";"#, Code),
                (r#"var s = $"{{call_the_target}}";"#, Str),
                (r#"char c = '"'; call_the_target();"#, Code),
            ],
        );
    }

    // ========================================================================
    // Kotlin Tests
    // ========================================================================

    #[test]
    fn kotlin_string_templates() {
        assert_regions(
            Language::Kotlin,
            &[
                (r#"val s = "Hello $call_the_target""#, Code),
                (r#"val s = "${call_the_target()}""#, Code),
                (r#"val s = "call_the_target""#, Str),
                (r#"val s = """a "// b""" + call_the_target()"#, Code),
                (r#"val u = "https://x"; call_the_target()"#, Code),
            ],
        );
    }

    // ========================================================================
    // Zig Tests
    // ========================================================================

    #[test]
    fn zig_strings_and_multiline_literals() {
        assert_regions(
            Language::Zig,
            &[
                (r#"const u = "https://x"; call_the_target();"#, Code),
                (r"    \\ call_the_target", Str),
                ("// call_the_target", Comment),
            ],
        );
    }

    #[test]
    fn non_ascii_text_before_the_pattern() {
        assert_regions(
            Language::Rust,
            &[
                (r#"let s = "héllo //"; call_the_target();"#, Code),
                ("let c = 'é'; call_the_target();", Code),
                ("// héllo call_the_target", Comment),
            ],
        );
    }

    #[test]
    fn text_tiers_have_no_filter() {
        for lang in [
            Language::Text,
            Language::Lock,
            Language::Generated,
            Language::Swift,
            Language::Unknown,
        ] {
            assert!(get_filter(lang).is_none(), "{lang:?}");
        }
    }
}
