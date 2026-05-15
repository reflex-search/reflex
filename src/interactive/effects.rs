use std::time::Duration;

/// Manager for animation effects
#[derive(Debug)]
pub struct EffectManager {
    /// Frame counter for animations (used for spinner rotation, etc.)
    frame_count: u64,
}

impl EffectManager {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }

    /// Update frame counter for animations
    pub fn update(&mut self, _elapsed: Duration) {
        // Increment frame counter for animations (used by spinner in UI)
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    /// Get current frame count for animations
    pub fn frame(&self) -> u64 {
        self.frame_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_manager_creation() {
        let manager = EffectManager::new();
        assert_eq!(manager.frame(), 0);
    }

    #[test]
    fn test_frame_increment() {
        let mut manager = EffectManager::new();
        assert_eq!(manager.frame(), 0);

        manager.update(Duration::from_millis(16));
        assert_eq!(manager.frame(), 1);

        manager.update(Duration::from_millis(16));
        assert_eq!(manager.frame(), 2);
    }
}
