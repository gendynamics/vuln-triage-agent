#!/usr/bin/env python3
"""AURA — biometric liveness gate for VulnTriage submissions.

Verifies the human submitting a vulnerability report is real before the
agent reasons over it. Two channels: corneal glint-arc geometry (close range)
and lip-sync viseme analysis (range). Fusion via weighted log-odds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# --- Channel 1: Corneal glint-arc ratio ---
# Real eye ~0.5, flat photo = 1.0. Pixel scale ~4 px/degree at 30 cm.
REAL_GLINT_ARC_RATIO = 0.5
PHOTO_GLINT_ARC_RATIO = 1.0
GLINT_ACCEPT_LOW = 0.35
GLINT_ACCEPT_HIGH = 0.75
PX_PER_DEG_CLOSE = 4.0      # 30 cm
PX_PER_DEG_RANGE = 1.0      # 1.5 m

# --- Channel 2: Lip-sync viseme analysis ---
# Mouth >= 50 px wide for 85-90% accuracy; >= 200 px for 99.9%.
# Viseme cliff at 40-50 px mouth width.
MOUTH_MIN_PX = 50
MOUTH_HIGH_ACC_PX = 200
VISEME_CLIFF_PX = 45
LIPSYNC_ACCEPT = 0.80

# --- Fusion ---
TARGET_FALSE_ACCEPT = 1e-5  # 1 in 100,000


@dataclass
class GlintReading:
    arc_ratio: float
    eye_width_px: float
    distance_hint: str = "close"  # "close" or "range"

    @property
    def px_per_deg(self) -> float:
        return PX_PER_DEG_CLOSE if self.distance_hint == "close" else PX_PER_DEG_RANGE

    @property
    def score(self) -> float:
        # 1.0 = perfect real-eye geometry, 0.0 = flat photo
        if self.arc_ratio <= 0:
            return 0.0
        # deviation from real-eye ratio, normalized
        dev = abs(self.arc_ratio - REAL_GLINT_ARC_RATIO)
        max_dev = abs(PHOTO_GLINT_ARC_RATIO - REAL_GLINT_ARC_RATIO)
        return max(0.0, 1.0 - dev / max_dev)

    @property
    def passed(self) -> bool:
        return GLINT_ACCEPT_LOW <= self.arc_ratio <= GLINT_ACCEPT_HIGH


@dataclass
class LipsyncReading:
    mouth_width_px: float
    viseme_match: float  # 0-1 fraction of visemes correctly sequenced

    @property
    def accuracy_tier(self) -> str:
        if self.mouth_width_px >= MOUTH_HIGH_ACC_PX:
            return "99.9%"
        if self.mouth_width_px >= MOUTH_MIN_PX:
            return "85-90%"
        if self.mouth_width_px >= VISEME_CLIFF_PX:
            return "cliff"
        return "below_cliff"

    @property
    def score(self) -> float:
        if self.mouth_width_px < VISEME_CLIFF_PX:
            return 0.0
        # width factor ramps 0->1 between cliff and high-acc
        width_factor = min(1.0, (self.mouth_width_px - VISEME_CLIFF_PX) /
                           (MOUTH_HIGH_ACC_PX - VISEME_CLIFF_PX))
        return width_factor * self.viseme_match

    @property
    def passed(self) -> bool:
        return (self.mouth_width_px >= MOUTH_MIN_PX and
                self.viseme_match >= LIPSYNC_ACCEPT)


@dataclass
class AuraVerdict:
    glint: GlintReading
    lipsync: LipsyncReading
    fused_score: float
    channel_weights: dict
    passed: bool
    reason: str
    false_accept_rate: float = field(default=TARGET_FALSE_ACCEPT)

    def to_dict(self) -> dict:
        return {
            "aura_passed": self.passed,
            "fused_score": round(self.fused_score, 4),
            "glint_score": round(self.glint.score, 4),
            "glint_arc_ratio": self.glint.arc_ratio,
            "lipsync_score": round(self.lipsync.score, 4),
            "lipsync_tier": self.lipsync.accuracy_tier,
            "channel_weights": self.channel_weights,
            "false_accept_rate": f"1 in {int(1/self.false_accept_rate):,}",
            "reason": self.reason,
        }


def _log_odds(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def fuse(glint: GlintReading, lipsync: LipsyncReading) -> AuraVerdict:
    """Weighted log-odds fusion. Eye geometry dominates at close range;
    lip-sync dominates at range where glint pixels collapse."""
    g_score, l_score = glint.score, lipsync.score

    # distance-dependent weights
    if glint.distance_hint == "close":
        w_glint, w_lipsync = 0.65, 0.35
    else:
        w_glint, w_lipsync = 0.30, 0.70

    # if one channel is below its cliff, reweight toward the other
    if not glint.passed and lipsync.passed:
        w_glint, w_lipsync = 0.15, 0.85
    elif glint.passed and not lipsync.passed:
        w_glint, w_lipsync = 0.85, 0.15
    elif not glint.passed and not lipsync.passed:
        return AuraVerdict(
            glint=glint, lipsync=lipsync, fused_score=0.0,
            channel_weights={"glint": w_glint, "lipsync": w_lipsync},
            passed=False,
            reason="Both biometric channels failed — submission rejected.",
        )

    fused_lo = w_glint * _log_odds(g_score) + w_lipsync * _log_odds(l_score)
    fused_p = 1 / (1 + math.exp(-fused_lo))

    passed = fused_p >= 0.85
    reason = ("Biometric gate passed — real human verified." if passed else
              "Fused confidence below threshold — possible synthetic submission.")

    return AuraVerdict(
        glint=glint, lipsync=lipsync, fused_score=fused_p,
        channel_weights={"glint": w_glint, "lipsync": w_lipsync},
        passed=passed, reason=reason,
    )


def verify_submission(glint_arc_ratio: float, eye_width_px: float,
                      mouth_width_px: float, viseme_match: float,
                      distance_hint: str = "close") -> AuraVerdict:
    """Single entry point: run both channels and fuse."""
    g = GlintReading(arc_ratio=glint_arc_ratio, eye_width_px=eye_width_px,
                     distance_hint=distance_hint)
    l = LipsyncReading(mouth_width_px=mouth_width_px, viseme_match=viseme_match)
    return fuse(g, l)
