#!/usr/bin/env rustc
// AURA-RS — Rust reference implementation of the biometric liveness gate.
// Mirrors the Python aura.py: corneal glint-arc ratio + lip-sync viseme fusion
// via weighted log-odds. This crate is the production-hardening path shown
// in the demo video's closing beat — same math, zero-cost abstractions,
// memory-safe by construction.

use serde::{Serialize, deserialize_from_str, Serialize as _}; // placeholder for build

/// Real-eye glint-arc ratio ~0.5; flat photo = 1.0.
pub const REAL_GLINT_ARC_RATIO: f64 = 0.5;
pub const PHOTO_GLINT_ARC_RATIO: f64 = 1.0;
pub const GLINT_ACCEPT_LOW: f64 = 0.35;
pub const GLINT_ACCEPT_HIGH: f64 = 0.75;

pub const MOUTH_MIN_PX: f64 = 50.0;
pub const MOUTH_HIGH_ACC_PX: f64 = 200.0;
pub const VISEME_CLIFF_PX: f64 = 45.0;
pub const LIPSYNC_ACCEPT: f64 = 0.80;

pub const TARGET_FALSE_ACCEPT: f64 = 1e-5; // 1 in 100,000

#[derive(Debug, Clone, Serialize)]
pub struct GlintReading {
    pub arc_ratio: f64,
    pub eye_width_px: f64,
    pub distance_hint: String,
}

impl GlintReading {
    pub fn score(&self) -> f64 {
        if self.arc_ratio <= 0.0 {
            return 0.0;
        }
        let dev = (self.arc_ratio - REAL_GLINT_ARC_RATIO).abs();
        let max_dev = (PHOTO_GLINT_ARC_RATIO - REAL_GLINT_ARC_RATIO).abs();
        (1.0 - dev / max_dev).max(0.0)
    }

    pub fn passed(&self) -> bool {
        self.arc_ratio >= GLINT_ACCEPT_LOW && self.arc_ratio <= GLINT_ACCEPT_HIGH
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LipsyncReading {
    pub mouth_width_px: f64,
    pub viseme_match: f64,
}

impl LipsyncReading {
    pub fn score(&self) -> f64 {
        if self.mouth_width_px < VISEME_CLIFF_PX {
            return 0.0;
        }
        let width_factor = ((self.mouth_width_px - VISEME_CLIFF_PX)
            / (MOUTH_HIGH_ACC_PX - VISEME_CLIFF_PX))
            .min(1.0);
        width_factor * self.viseme_match
    }

    pub fn passed(&self) -> bool {
        self.mouth_width_px >= MOUTH_MIN_PX && self.viseme_match >= LIPSYNC_ACCEPT
    }
}

fn log_odds(p: f64) -> f64 {
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    (p / (1.0 - p)).ln()
}

/// Weighted log-odds fusion. Eye geometry dominates at close range;
/// lip-sync dominates at range where glint pixels collapse.
pub fn fuse(glint: &GlintReading, lipsync: &LipsyncReading) -> (f64, bool, String) {
    let (w_glint, w_lipsync) = if !glint.passed() && lipsync.passed() {
        (0.15, 0.85)
    } else if glint.passed() && !lipsync.passed() {
        (0.85, 0.15)
    } else if !glint.passed() && !lipsync.passed() {
        return (0.0, false, "Both biometric channels failed".into());
    } else if glint.distance_hint == "close" {
        (0.65, 0.35)
    } else {
        (0.30, 0.70)
    };

    let fused_lo = w_glint * log_odds(glint.score()) + w_lipsync * log_odds(lipsync.score());
    let fused_p = 1.0 / (1.0 + (-fused_lo).exp());
    let passed = fused_p >= 0.85;
    let reason = if passed {
        "Biometric gate passed — real human verified.".into()
    } else {
        "Fused confidence below threshold — possible synthetic submission.".into()
    };
    (fused_p, passed, reason)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_eye_close_range_passes() {
        let g = GlintReading { arc_ratio: 0.52, eye_width_px: 120.0, distance_hint: "close".into() };
        let l = LipsyncReading { mouth_width_px: 120.0, viseme_match: 0.92 };
        let (score, passed, _) = fuse(&g, &l);
        assert!(passed, "expected pass, got score {:.4}", score);
    }

    #[test]
    fn flat_photo_fails() {
        let g = GlintReading { arc_ratio: 1.0, eye_width_px: 120.0, distance_hint: "close".into() };
        let l = LipsyncReading { mouth_width_px: 120.0, viseme_match: 0.92 };
        let (_, passed, _) = fuse(&g, &l);
        assert!(!passed);
    }

    #[test]
    fn range_strong_lipsync_passes() {
        let g = GlintReading { arc_ratio: 0.9, eye_width_px: 30.0, distance_hint: "range".into() };
        let l = LipsyncReading { mouth_width_px: 220.0, viseme_match: 0.95 };
        let (score, passed, _) = fuse(&g, &l);
        assert!(passed, "expected pass, got score {:.4}", score);
    }
}
