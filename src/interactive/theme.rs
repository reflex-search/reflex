use ratatui::style::{Color, Style};
use std::env;
use syntect::highlighting::{Theme, ThemeSet};

/// Theme manager for syntax highlighting and UI colors
#[derive(Debug, Clone)]
pub struct ThemeManager {
    /// Background type (dark or light)
    pub background: BackgroundType,
    /// Syntect theme for code highlighting
    pub syntax_theme: String,
    /// UI color palette
    pub palette: ColorPalette,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BackgroundType {
    Dark,
    Light,
}

/// Color palette for UI elements
#[derive(Debug, Clone)]
pub struct ColorPalette {
    // Status colors
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub info: Color,

    // UI element colors
    pub accent: Color,
    pub highlight: Color,
    pub muted: Color,
    pub background: Color,
    pub foreground: Color,

    // Filter badge colors
    pub badge_active: Color,
    pub badge_inactive: Color,
}

impl ThemeManager {
    /// Detect theme from terminal environment
    pub fn detect() -> Self {
        let background = Self::detect_background();
        let syntax_theme = Self::get_syntax_theme(&background);
        let palette = ColorPalette::for_background(&background);

        Self {
            background,
            syntax_theme,
            palette,
        }
    }

    fn detect_background() -> BackgroundType {
        // Parse COLORFGBG environment variable
        // Format: "foreground;background" where 0-7=dark, 8-15=light
        if let Ok(colorfgbg) = env::var("COLORFGBG") {
            if let Some(bg) = colorfgbg.split(';').nth(1) {
                if let Ok(bg_val) = bg.parse::<u8>() {
                    return if bg_val < 8 {
                        BackgroundType::Dark
                    } else {
                        BackgroundType::Light
                    };
                }
            }
        }

        // Default to dark if unable to detect
        BackgroundType::Dark
    }

    fn get_syntax_theme(background: &BackgroundType) -> String {
        match background {
            BackgroundType::Dark => "Monokai Extended".to_string(),
            BackgroundType::Light => "InspiredGitHub".to_string(),
        }
    }

    /// Load the syntect theme
    pub fn load_syntect_theme(&self) -> Theme {
        let theme_set = ThemeSet::load_defaults();

        // Try to load the preferred theme, fall back to a default
        if let Some(theme) = theme_set.themes.get(&self.syntax_theme) {
            theme.clone()
        } else {
            // Fallback to base16 themes if our preferred themes aren't available
            match self.background {
                BackgroundType::Dark => theme_set
                    .themes
                    .get("base16-ocean.dark")
                    .or_else(|| theme_set.themes.values().next())
                    .cloned()
                    .expect("No themes available"),
                BackgroundType::Light => theme_set
                    .themes
                    .get("base16-ocean.light")
                    .or_else(|| theme_set.themes.values().next())
                    .cloned()
                    .expect("No themes available"),
            }
        }
    }
}

impl ColorPalette {
    pub fn for_background(bg: &BackgroundType) -> Self {
        match bg {
            BackgroundType::Dark => Self {
                success: Color::Green,
                warning: Color::Yellow,
                error: Color::Red,
                info: Color::Cyan,
                accent: Color::Magenta,
                highlight: Color::LightBlue,
                muted: Color::DarkGray,
                background: Color::Black,
                foreground: Color::White,
                badge_active: Color::Cyan,
                badge_inactive: Color::DarkGray,
            },
            BackgroundType::Light => Self {
                success: Color::Green,
                warning: Color::Yellow,
                error: Color::Red,
                info: Color::Blue,
                accent: Color::Magenta,
                highlight: Color::Blue,
                muted: Color::Gray,
                background: Color::White,
                foreground: Color::Black,
                badge_active: Color::Blue,
                badge_inactive: Color::Gray,
            },
        }
    }

    /// Get a style for success messages
    pub fn success_style(&self) -> Style {
        Style::default().fg(self.success)
    }

    /// Get a style for warnings
    pub fn warning_style(&self) -> Style {
        Style::default().fg(self.warning)
    }

    /// Get a style for errors
    pub fn error_style(&self) -> Style {
        Style::default().fg(self.error)
    }

    /// Get a style for highlighted text
    pub fn highlight_style(&self) -> Style {
        Style::default().fg(self.highlight)
    }

    /// Get a style for muted text
    pub fn muted_style(&self) -> Style {
        Style::default().fg(self.muted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theme_detection() {
        let theme = ThemeManager::detect();
        // Should not panic and should have valid values
        assert!(!theme.syntax_theme.is_empty());
    }

    #[test]
    fn test_palette_creation() {
        let palette = ColorPalette::for_background(&BackgroundType::Dark);
        assert_eq!(palette.success, Color::Green);

        let palette = ColorPalette::for_background(&BackgroundType::Light);
        assert_eq!(palette.success, Color::Green);
    }
}
