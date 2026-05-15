//! Syntax highlighting utilities for interactive mode
//!
//! This module provides syntax highlighting for code previews in the TUI,
//! converting syntect highlighting to ratatui Spans.

use ratatui::{
    style::{Color, Style},
    text::{Line, Span},
};
use std::sync::OnceLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Color as SyntectColor, Style as SyntectStyle, Theme, ThemeSet};
use syntect::parsing::{SyntaxReference, SyntaxSet};

use crate::models::Language;

/// Lazy-loaded syntax highlighting resources
struct SyntaxHighlighter {
    syntax_set: SyntaxSet,
    theme_set: ThemeSet,
}

impl SyntaxHighlighter {
    fn new() -> Self {
        Self {
            syntax_set: SyntaxSet::load_defaults_newlines(),
            theme_set: ThemeSet::load_defaults(),
        }
    }

    /// Get syntax reference for a language using extension-based lookup (most reliable)
    ///
    /// This uses file extensions to find syntaxes, which is more reliable than name-based
    /// lookup because syntect (based on Sublime Text) primarily uses extension matching.
    ///
    /// For languages not in the default syntect set (TypeScript, Vue, Svelte), we fall back
    /// to related syntaxes (JavaScript for TypeScript, HTML for Vue/Svelte).
    fn get_syntax(&self, lang: &Language) -> Option<&SyntaxReference> {
        let (extension, fallback_extension) = match lang {
            Language::Rust => ("rs", None),
            Language::Python => ("py", None),
            Language::JavaScript => ("js", None),
            Language::TypeScript => ("ts", Some("js")), // Fallback to JavaScript
            Language::Go => ("go", None),
            Language::Java => ("java", None),
            Language::C => ("c", None),
            Language::Cpp => ("cpp", None),
            Language::CSharp => ("cs", None),
            Language::PHP => ("php", None),
            Language::Ruby => ("rb", None),
            Language::Kotlin => ("kt", None),
            Language::Swift => ("swift", None),
            Language::Zig => ("zig", None),
            Language::Vue => ("vue", Some("html")), // Fallback to HTML
            Language::Svelte => ("svelte", Some("html")), // Fallback to HTML
            Language::Unknown => return None,
        };

        // Try extension-based lookup first (most reliable)
        self.syntax_set
            .find_syntax_by_extension(extension)
            .or_else(|| {
                // Try token-based search (searches by extension then name)
                self.syntax_set.find_syntax_by_token(extension)
            })
            .or_else(|| {
                // If we have a fallback extension (for TypeScript, Vue, Svelte), try it
                fallback_extension.and_then(|fallback| {
                    self.syntax_set
                        .find_syntax_by_extension(fallback)
                        .or_else(|| self.syntax_set.find_syntax_by_token(fallback))
                })
            })
    }
}

// Global syntax highlighter (initialized on first use)
static SYNTAX_HIGHLIGHTER: OnceLock<SyntaxHighlighter> = OnceLock::new();

fn get_syntax_highlighter() -> &'static SyntaxHighlighter {
    SYNTAX_HIGHLIGHTER.get_or_init(SyntaxHighlighter::new)
}

/// Convert syntect color to ratatui color
fn syntect_color_to_ratatui(color: SyntectColor) -> Color {
    Color::Rgb(color.r, color.g, color.b)
}

/// Highlight multiple lines of code and return a vector of Lines with colored Spans
///
/// This function maintains syntect state across lines to properly handle multi-line
/// constructs like strings, comments, and code blocks.
///
/// If the language is not supported or highlighting fails, returns plain text lines.
pub fn highlight_code_lines<'a>(lines: &[String], lang: Language, theme: &Theme) -> Vec<Line<'a>> {
    // Early return for unknown languages
    if !lang.is_supported() {
        return lines
            .iter()
            .map(|line| Line::from(line.to_string()))
            .collect();
    }

    let highlighter = get_syntax_highlighter();

    // Get syntax for the language
    let syntax = match highlighter.get_syntax(&lang) {
        Some(s) => s,
        None => {
            return lines
                .iter()
                .map(|line| Line::from(line.to_string()))
                .collect();
        }
    };

    // Create highlighter instance that maintains state across lines
    let mut h = HighlightLines::new(syntax, theme);
    let mut result = Vec::with_capacity(lines.len());

    for line in lines {
        // Highlight the line (state is maintained in h)
        let ranges: Vec<(SyntectStyle, &str)> =
            match h.highlight_line(line, &highlighter.syntax_set) {
                Ok(ranges) => ranges,
                Err(_) => {
                    // On error, fall back to plain text for this line
                    result.push(Line::from(line.to_string()));
                    continue;
                }
            };

        // Convert syntect ranges to ratatui Spans
        let spans: Vec<Span<'a>> = ranges
            .into_iter()
            .map(|(style, text)| {
                let fg = syntect_color_to_ratatui(style.foreground);
                let ratatui_style = Style::default().fg(fg);
                Span::styled(text.to_string(), ratatui_style)
            })
            .collect();

        result.push(Line::from(spans));
    }

    result
}

/// Get the default theme based on terminal background
///
/// Returns Monokai Extended for dark terminals, InspiredGitHub for light terminals.
pub fn get_default_theme(is_dark: bool) -> Theme {
    let highlighter = get_syntax_highlighter();

    let theme_name = if is_dark {
        "Monokai Extended"
    } else {
        "InspiredGitHub"
    };

    highlighter
        .theme_set
        .themes
        .get(theme_name)
        .or_else(|| {
            // Fallback to base16 themes if preferred themes aren't available
            let fallback_name = if is_dark {
                "base16-ocean.dark"
            } else {
                "base16-ocean.light"
            };
            highlighter.theme_set.themes.get(fallback_name)
        })
        .or_else(|| highlighter.theme_set.themes.values().next())
        .cloned()
        .expect("No themes available in syntect")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlighter_initialization() {
        let highlighter = get_syntax_highlighter();

        // Core languages that should work
        assert!(highlighter.get_syntax(&Language::Rust).is_some());
        assert!(highlighter.get_syntax(&Language::Python).is_some());
        assert!(highlighter.get_syntax(&Language::JavaScript).is_some());

        // Languages with fallbacks
        assert!(
            highlighter.get_syntax(&Language::TypeScript).is_some(),
            "TypeScript should fallback to JavaScript"
        );
        assert!(
            highlighter.get_syntax(&Language::Vue).is_some(),
            "Vue should fallback to HTML"
        );
        assert!(
            highlighter.get_syntax(&Language::Svelte).is_some(),
            "Svelte should fallback to HTML"
        );

        // Unknown should return None
        assert!(highlighter.get_syntax(&Language::Unknown).is_none());
    }

    #[test]
    fn test_highlight_lines_rust() {
        let lines = vec![
            "fn main() {".to_string(),
            "    println!(\"Hello\");".to_string(),
            "}".to_string(),
        ];
        let theme = get_default_theme(true);
        let highlighted = highlight_code_lines(&lines, Language::Rust, &theme);

        // Should return same number of lines
        assert_eq!(highlighted.len(), 3);

        // Each line should have spans (syntax highlighting)
        for line in &highlighted {
            assert!(!line.spans.is_empty());
        }
    }

    #[test]
    fn test_highlight_lines_unknown_language() {
        let lines = vec!["some code".to_string(), "more code".to_string()];
        let theme = get_default_theme(true);
        let highlighted = highlight_code_lines(&lines, Language::Unknown, &theme);

        // Should return plain text for unknown languages
        assert_eq!(highlighted.len(), 2);
        for line in &highlighted {
            assert_eq!(line.spans.len(), 1);
        }
    }

    #[test]
    fn test_highlight_lines_maintains_state() {
        // Test that multi-line constructs like strings work properly
        let lines = vec![
            "<?php".to_string(),
            "use Some\\Namespace;".to_string(),
            "use Another\\Namespace;".to_string(),
        ];
        let theme = get_default_theme(true);
        let highlighted = highlight_code_lines(&lines, Language::PHP, &theme);

        // Should return same number of lines
        assert_eq!(highlighted.len(), 3);

        // All lines should have syntax highlighting
        for line in &highlighted {
            assert!(!line.spans.is_empty());
        }
    }

    #[test]
    fn test_color_conversion() {
        let syntect_color = SyntectColor {
            r: 255,
            g: 128,
            b: 64,
            a: 255,
        };
        let ratatui_color = syntect_color_to_ratatui(syntect_color);

        match ratatui_color {
            Color::Rgb(r, g, b) => {
                assert_eq!(r, 255);
                assert_eq!(g, 128);
                assert_eq!(b, 64);
            }
            _ => panic!("Expected RGB color"),
        }
    }

    #[test]
    fn test_get_default_theme() {
        let dark_theme = get_default_theme(true);
        let light_theme = get_default_theme(false);

        // Themes should be different
        assert_ne!(dark_theme.name.as_ref(), light_theme.name.as_ref());
    }

    #[test]
    fn test_all_supported_languages_have_syntax() {
        let highlighter = get_syntax_highlighter();
        let theme = get_default_theme(true);

        // Test ALL supported languages (except Swift which is temporarily disabled)
        let all_languages = vec![
            (Language::Rust, "fn main() {}"),
            (Language::Python, "def main():"),
            (Language::JavaScript, "function main() {}"),
            (Language::TypeScript, "const x: string = '';"),
            (Language::Go, "func main() {}"),
            (Language::Java, "public class Main {}"),
            (Language::C, "int main() {}"),
            (Language::Cpp, "int main() {}"),
            (Language::CSharp, "public class Main {}"),
            (Language::PHP, "<?php function main() {}"),
            (Language::Ruby, "def main; end"),
            (Language::Kotlin, "fun main() {}"),
            (Language::Zig, "pub fn main() void {}"),
            (Language::Vue, "<template></template>"),
            (Language::Svelte, "<script></script>"),
        ];

        for (lang, code) in all_languages {
            let lines = vec![code.to_string()];
            let highlighted = highlight_code_lines(&lines, lang, &theme);

            // Should return highlighted content (not plain text fallback)
            assert_eq!(highlighted.len(), 1, "Failed for {:?}", lang);
            assert!(
                !highlighted[0].spans.is_empty(),
                "{:?} has no syntax highlighting",
                lang
            );
        }
    }
}
