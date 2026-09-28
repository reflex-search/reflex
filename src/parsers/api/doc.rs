//! Doc comment parsing: markdown in, summary / sections / examples / links out.
//!
//! Rustdoc conventions: a `# Heading` starts a section (`# Errors`, `# Panics`,
//! `# Safety`, `# Examples`); a bare or `rust` fence is a doctest, and its lines that
//! start with `# ` are hidden from readers.

use super::{CodeExample, DocComment};
use std::sync::LazyLock;

/// Build a [`DocComment`] from comment lines with their markers already removed.
pub fn from_lines(lines: &[String], start_line: u32, end_line: u32) -> Option<DocComment> {
    let markdown = dedent(lines).join("\n").trim_matches('\n').to_string();
    if markdown.trim().is_empty() {
        return None;
    }
    Some(DocComment {
        summary: summary(&markdown),
        sections: sections(&markdown),
        examples: examples(&markdown),
        links: links(&markdown),
        markdown,
        start_line,
        end_line,
    })
}

/// Remove the common leading indentation (usually the one space after `///`).
fn dedent(lines: &[String]) -> Vec<String> {
    let indent = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start_matches(' ').len())
        .min()
        .unwrap_or(0);
    lines
        .iter()
        .map(|l| {
            if l.len() >= indent && l.is_char_boundary(indent) {
                l[indent..].trim_end().to_string()
            } else {
                l.trim().to_string()
            }
        })
        .collect()
}

fn is_fence(line: &str) -> Option<&str> {
    let t = line.trim_start();
    t.strip_prefix("```").or_else(|| t.strip_prefix("~~~"))
}

/// The first paragraph, outside fences, stopping at a heading.
pub fn summary(markdown: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for line in markdown.lines() {
        let t = line.trim();
        if is_fence(line).is_some() || t.starts_with('#') {
            break;
        }
        if t.is_empty() {
            if out.is_empty() {
                continue;
            }
            break;
        }
        out.push(t);
    }
    out.join(" ")
}

/// `# Heading` sections (level 1 and 2), outside fences: (lowercase heading, body).
pub fn sections(markdown: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, Vec<&str>)> = Vec::new();
    let mut in_fence = false;
    for line in markdown.lines() {
        if is_fence(line).is_some() {
            in_fence = !in_fence;
        }
        let t = line.trim_start();
        let heading = (!in_fence)
            .then(|| t.strip_prefix("# ").or_else(|| t.strip_prefix("## ")))
            .flatten();
        match heading {
            Some(h) => out.push((h.trim().to_ascii_lowercase(), Vec::new())),
            None => {
                if let Some((_, body)) = out.last_mut() {
                    body.push(line);
                }
            }
        }
    }
    out.into_iter()
        .map(|(h, body)| (h, body.join("\n").trim_matches('\n').to_string()))
        .collect()
}

/// Code fences. Rust fences drop rustdoc's hidden `# ` lines.
pub fn examples(markdown: &str) -> Vec<CodeExample> {
    let mut out = Vec::new();
    let mut current: Option<(String, bool, Vec<String>)> = None;
    for line in markdown.lines() {
        match (is_fence(line), current.take()) {
            (Some(info), None) => {
                let (lang, doctest) = classify_fence(info.trim());
                current = Some((lang, doctest, Vec::new()));
            }
            (Some(_), Some((lang, doctest, body))) => out.push(CodeExample {
                code: body.join("\n"),
                lang,
                doctest,
            }),
            (None, Some((lang, doctest, mut body))) => {
                let t = line.trim_start();
                let hidden = lang == "rust" && (t == "#" || t.starts_with("# "));
                if !hidden {
                    // `##` escapes a literal `#` at the start of a line.
                    let shown = if lang == "rust" && t.starts_with("##") {
                        line.replacen("##", "#", 1)
                    } else {
                        line.to_string()
                    };
                    body.push(shown);
                }
                current = Some((lang, doctest, body));
            }
            (None, None) => {}
        }
    }
    out
}

/// Fence info string → (language, is a doctest).
fn classify_fence(info: &str) -> (String, bool) {
    let tags: Vec<&str> = info
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();
    const RUSTDOC_ATTRS: &[&str] = &[
        "rust",
        "ignore",
        "no_run",
        "should_panic",
        "compile_fail",
        "edition2015",
        "edition2018",
        "edition2021",
        "edition2024",
        "test_harness",
        "standalone_crate",
    ];
    let rusty = tags.is_empty()
        || tags
            .iter()
            .all(|t| RUSTDOC_ATTRS.contains(t) || t.starts_with("edition"));
    if rusty {
        let skipped = tags.iter().any(|t| *t == "ignore" || *t == "compile_fail");
        ("rust".into(), !skipped)
    } else {
        (tags[0].to_string(), false)
    }
}

static LINK_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    // [`Foo`] / [Foo] (not followed by `(` or `[`) / [text](crate::x::Foo)
    regex::Regex::new(r"\[`?([A-Za-z_][\w:]*(?:\(\))?)`?\](?:[^(\[]|$)|\]\(([A-Za-z_][\w]*(?:::[\w]+)*(?:\(\))?)\)")
        .expect("valid regex")
});

/// Intra-doc link targets, as written (`Foo`, `crate::x::Bar`, `Self::new`), deduped.
pub fn links(markdown: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut in_fence = false;
    for line in markdown.lines() {
        if is_fence(line).is_some() {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        for cap in LINK_RE.captures_iter(line) {
            if let Some(m) = cap.get(1).or_else(|| cap.get(2)) {
                let target = m.as_str().trim_end_matches("()").to_string();
                if !out.contains(&target) {
                    out.push(target);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(s: &str) -> DocComment {
        let lines: Vec<String> = s.lines().map(|l| format!(" {l}")).collect();
        from_lines(&lines, 1, lines.len() as u32).unwrap()
    }

    #[test]
    fn summary_sections_examples() {
        let d = doc(
            "Adds two numbers,\nsaturating on overflow.\n\nMore detail.\n\n# Errors\n\nNever.\n\n# Examples\n\n```\n# use demo::add;\nassert_eq!(add(1, 2), 3);\n## not hidden\n```\n\n```text\nplain\n```",
        );
        assert_eq!(d.summary, "Adds two numbers, saturating on overflow.");
        assert_eq!(d.sections[0], ("errors".into(), "Never.".into()));
        assert_eq!(d.sections[1].0, "examples");
        assert_eq!(d.examples.len(), 2);
        assert_eq!(d.examples[0].lang, "rust");
        assert!(d.examples[0].doctest);
        assert_eq!(
            d.examples[0].code,
            "assert_eq!(add(1, 2), 3);\n# not hidden"
        );
        assert_eq!(d.examples[1].lang, "text");
        assert!(!d.examples[1].doctest);
        assert!(
            !d.markdown.starts_with(' '),
            "one-space marker indent removed"
        );
    }

    #[test]
    fn fences_classify() {
        assert_eq!(classify_fence(""), ("rust".into(), true));
        assert_eq!(classify_fence("rust,no_run"), ("rust".into(), true));
        assert_eq!(classify_fence("ignore"), ("rust".into(), false));
        assert_eq!(classify_fence("compile_fail"), ("rust".into(), false));
        assert_eq!(classify_fence("toml"), ("toml".into(), false));
    }

    #[test]
    fn intra_doc_links() {
        let d = doc(
            "See [`QueryEngine`] and [Self::new], or [the cache](crate::cache::CacheManager).\n[not a link](https://example.com)\n```\n[`InCode`]\n```",
        );
        assert_eq!(
            d.links,
            vec!["QueryEngine", "Self::new", "crate::cache::CacheManager"]
        );
    }

    #[test]
    fn heading_in_fence_is_not_a_section() {
        let d = doc("Summary.\n\n```\n# hidden setup\n```\n");
        assert!(d.sections.is_empty());
    }

    #[test]
    fn empty_doc_is_none() {
        assert!(from_lines(&[" ".into(), String::new()], 1, 2).is_none());
    }
}
