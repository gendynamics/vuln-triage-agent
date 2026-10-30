#!/usr/bin/env python3
"""AURA silent scan — pure-Python mirror of crates/aura-rs/src/scan.rs.

Scores a 5-second window of per-frame face landmarks + grayscale eye crops
(from an upstream detector) into a liveness verdict using evidence
accumulation (summed, reliability-weighted log-likelihood ratios) plus hard
spoof vetoes. Renders nothing. Results carry scores and reason codes only:
no pixels, no landmarks, no biometric template. Liveness, NOT identity.

All thresholds and likelihood parameters are hand-set and UNCALIBRATED. The
0.999 high-assurance tier and the 1e-5 false-accept target are design goals,
not measured accuracy; validate on a real labelled anti-spoofing dataset.

`capture_and_scan()` is an optional headless webcam adapter (OpenCV +
MediaPipe Face Mesh, imported lazily). It opens no window and draws nothing;
the OS camera-in-use indicator and permission prompt remain in effect.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple

from aura import (GlintReading, LipsyncReading, fuse, LIPSYNC_ACCEPT,
                  TARGET_FALSE_ACCEPT, VISEME_CLIFF_PX, MOUTH_HIGH_ACC_PX,
                  GLINT_ACCEPT_HIGH)

Pt = Tuple[float, float]

WINDOW_MS = 5000
MIN_FRAMES = 30
MIN_SPAN_MS = 3000
CLOSE_EYE_WIDTH_PX = 60.0
MIN_IRIS_MOTION = 0.01
MIN_LOCK_FRACTION = 0.6
STATIC_MESH_EPS = 0.004
MAX_DUP_FRACTION = 0.10
GLINT_MIN_LEVEL = 180
FLAT_GLINT_RATIO = 0.85
PLANAR_RESIDUAL_EPS = 0.003
JITTER_MIN, JITTER_MAX = 0.0015, 0.06
REF_DT_MS = 1000.0 / 30.0
MOIRE_ROW_SCORE = 0.5
FLICKER_ALT_FRACTION = 0.6
FLICKER_MIN_STEP = 1.5
AV_MAX_LAG_MS = 120.0
AV_MIN_CORR = 0.3
LIGHT_MIN_COS = 0.5
LEGACY_THRESHOLD = 0.85
HIGH_ASSURANCE_THRESHOLD = 0.999

VETOES = {
    "TimestampNonMonotonic", "DuplicateFrames", "StaticMesh",
    "FlatGlintGeometry", "PlanarRigidMotion", "MoirePattern", "ScreenFlicker",
    "RectangularGlint", "DoubledGlint", "AvDesync", "JitterOutOfRange",
    "GlintLightMismatch", "ChallengeFailed",
}


def llr_accept_threshold() -> float:
    """Wald bound: accept only if LR >= 1/alpha  =>  P(accept|spoof) <= alpha
    (assuming calibrated, conditionally independent channels)."""
    return math.log(1.0 / TARGET_FALSE_ACCEPT)


@dataclass
class EyePatch:
    origin: Pt
    width: int
    height: int
    pixels: bytearray  # row-major uint8


@dataclass
class EyeObs:
    contour: List[Pt]
    iris_center: Pt
    iris_radius: float
    patch: EyePatch


@dataclass
class FrameObs:
    t_ms: int
    left_eye: EyeObs
    right_eye: EyeObs
    lips_outer: List[Pt]
    lips_inner: List[Pt]
    audio_openness: Optional[float] = None


@dataclass
class Calibration:
    """Platt/isotonic map from accumulated LLR to probability.
    Defaults are identity and explicitly UNCALIBRATED."""
    platt_a: float = 1.0
    platt_b: float = 0.0
    isotonic: List[Tuple[float, float]] = field(default_factory=list)
    calibrated: bool = False
    prior_log_odds: float = 0.0

    @classmethod
    def from_config(cls, text: str) -> "Calibration":
        c = cls()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"bad line: {line}")
            k, v = (s.strip() for s in line.split("=", 1))
            if k in ("platt_a", "platt_b", "prior_log_odds"):
                setattr(c, k, float(v))
            elif k == "calibrated":
                c.calibrated = v == "true"
            elif k == "isotonic":
                c.isotonic = [tuple(float(x) for x in p.split(":")) for p in v.split(",")]
        return c

    def posterior(self, llr_total: float) -> float:
        z = self.platt_a * (llr_total + self.prior_log_odds) + self.platt_b
        p = 1 / (1 + math.exp(-z))
        t = self.isotonic
        if len(t) < 2:
            return p
        if p <= t[0][0]:
            return t[0][1]
        for (x0, y0), (x1, y1) in zip(t, t[1:]):
            if p <= x1:
                return y0 + (p - x0) / max(x1 - x0, 1e-12) * (y1 - y0)
        return t[-1][1]


@dataclass
class ScanResult:
    status: str  # "pass" | "fail" | "inconclusive"
    tier: str = "high_assurance"
    fused_score: float = 0.0
    legacy_pass: bool = False
    posterior: float = 0.0
    high_assurance_pass: bool = False
    calibrated: bool = False
    evidence: dict = field(default_factory=dict)
    glint_score: float = 0.0
    glint_arc_ratio: float = 0.0
    lipsync_score: float = 0.0
    viseme_match: float = 0.0
    distance_hint: str = "unknown"
    frames_used: int = 0
    glint_lock_fraction: Tuple[float, float] = (0.0, 0.0)
    vetoes: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["aura_passed"] = self.passed
        return d


def _early(status, n, code):
    return ScanResult(status=status, frames_used=n, reasons=[code],
                      vetoes=[code] if code in VETOES else [])


# ---------- helpers ----------
def _centroid(p):
    n = max(len(p), 1)
    return (sum(q[0] for q in p) / n, sum(q[1] for q in p) / n)


def _extent(p, y=False):
    if not p:
        return 0.0
    v = [q[1] if y else q[0] for q in p]
    return max(v) - min(v)


def _area(p):
    n = len(p)
    if n < 3:
        return 0.0
    return abs(sum(p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1]
                   for i in range(n))) * 0.5


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _std(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _median(v):
    return sorted(v)[len(v) // 2] if v else 0.0


def _sign(x):
    return -1.0 if x < 0 else 1.0  # matches Rust f64::signum (0 -> +1)


def _pearson_lag(a, b, lag):
    xa, xb = [], []
    for i in range(len(a)):
        j = i + lag
        if 0 <= j < len(a):
            xa.append(a[i]); xb.append(b[j])
    ma, mb = _mean(xa), _mean(xb)
    num = sum((x - ma) * (y - mb) for x, y in zip(xa, xb))
    da = sum((x - ma) ** 2 for x in xa)
    db = sum((y - mb) ** 2 for y in xb)
    return 0.0 if da <= 0 or db <= 0 else num / math.sqrt(da * db)


def detect_glint(eye: EyeObs):
    """Return (glint_xy or None, components, area, bbox_fill)."""
    p = eye.patch
    if len(p.pixels) != p.width * p.height or eye.iris_radius <= 0:
        return None, 0, 0, 0.0
    cx, cy = eye.iris_center[0] - p.origin[0], eye.iris_center[1] - p.origin[1]
    r2 = eye.iris_radius ** 2
    x_lo, x_hi = max(0, int(cx - eye.iris_radius) - 1), min(p.width, int(cx + eye.iris_radius) + 2)
    y_lo, y_hi = max(0, int(cy - eye.iris_radius) - 1), min(p.height, int(cy + eye.iris_radius) + 2)
    inside = [(x, y) for y in range(y_lo, y_hi) for x in range(x_lo, x_hi)
              if (x - cx) ** 2 + (y - cy) ** 2 <= r2]
    if not inside:
        return None, 0, 0, 0.0
    px, w = p.pixels, p.width
    peak = max(px[y * w + x] for x, y in inside)
    if peak < GLINT_MIN_LEVEL:
        return None, 0, 0, 0.0
    thr = max(peak * 0.85, GLINT_MIN_LEVEL)
    hits = [(x, y, px[y * w + x]) for x, y in inside if px[y * w + x] >= thr]
    if not hits or len(hits) > 0.15 * len(inside):
        return None, 0, 0, 0.0
    sw = sum(v for _, _, v in hits)
    gx = sum(v * x for x, _, v in hits) / sw + p.origin[0]
    gy = sum(v * y for _, y, v in hits) / sw + p.origin[1]
    mask = {(x, y) for x, y, _ in hits}
    seen, comps = set(), 0
    for s in mask:
        if s in seen:
            continue
        stack, size = [s], 0
        seen.add(s)
        while stack:
            x, y = stack.pop(); size += 1
            for nb in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nb in mask and nb not in seen:
                    seen.add(nb); stack.append(nb)
        if size >= 2:
            comps += 1
    xs, ys = [h[0] for h in hits], [h[1] for h in hits]
    bbox = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
    return (gx, gy), comps, len(hits), len(hits) / bbox


def build_lip_mesh(outer: Sequence[Pt], inner: Sequence[Pt]) -> dict:
    n = min(len(outer), len(inner))
    verts = list(outer[:n]) + list(inner[:n])
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(i, j, n + i), (j, n + j, n + i)]
    w = _extent(outer)
    return {"vertices": verts, "triangles": tris, "width_px": w,
            "openness": _area(inner) / (w * w) if w > 0 else 0.0,
            "spread": _extent(inner) / w if w > 0 else 0.0}


def _sig(f: FrameObs):
    pts = []
    for e in (f.left_eye, f.right_eye):
        pts += e.contour + [e.iris_center]
    return hash((tuple(pts), bytes(f.left_eye.patch.pixels), bytes(f.right_eye.patch.pixels),
                 tuple(f.lips_outer), tuple(f.lips_inner)))


def _articulation(open_, spread, t_s):
    n = len(open_)
    if n < 3:
        return 0.0
    rng = max(open_) - min(open_)
    range_s = min(rng / 0.06, 1.0)
    m = _mean(open_)
    cross = sum(1 for i in range(1, n) if _sign(open_[i - 1] - m) != _sign(open_[i] - m))
    cps = cross / 2.0 / max(t_s[-1] - t_s[0], 1e-6)
    rate_s = 1.0 if 0.5 <= cps <= 8.0 else 0.3
    jumps = sum(1 for i in range(1, n) if abs(open_[i] - open_[i - 1]) > 0.5 * max(rng, 1e-9))
    smooth_s = 1.0 - min(jumps / n, 1.0)
    spread_s = 1.0 if (_std(spread) > 1e-4 or rng < 1e-6) else 0.6
    return min(max(0.5 * range_s + 0.2 * rate_s + 0.2 * smooth_s + 0.1 * spread_s, 0.0), 1.0)


def _det3(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def _affine_residual(p0, pk):
    n = min(len(p0), len(pk))
    if n < 4:
        return math.inf
    m = [[0.0] * 3 for _ in range(3)]
    bx, by = [0.0] * 3, [0.0] * 3
    for i in range(n):
        v = (p0[i][0], p0[i][1], 1.0)
        for r in range(3):
            for c in range(3):
                m[r][c] += v[r] * v[c]
            bx[r] += v[r] * pk[i][0]; by[r] += v[r] * pk[i][1]
    d = _det3(m)
    if abs(d) < 1e-12:
        return math.inf

    def solve(b):
        out = []
        for c in range(3):
            a = [row[:] for row in m]
            for r in range(3):
                a[r][c] = b[r]
            out.append(_det3(a) / d)
        return out
    sx, sy = solve(bx), solve(by)
    e = 0.0
    for i in range(n):
        x, y = p0[i]
        e += (sx[0] * x + sx[1] * y + sx[2] - pk[i][0]) ** 2 + (sy[0] * x + sy[1] * y + sy[2] - pk[i][1]) ** 2
    return math.sqrt(e / n)


def _landmarks(f):
    v = []
    for e in (f.left_eye, f.right_eye):
        v += e.contour + [e.iris_center]
    return v + list(f.lips_outer) + list(f.lips_inner)


def _moire(p: EyePatch):
    if len(p.pixels) != p.width * p.height or p.width < 20:
        return None
    scores = []
    for y in range(p.height):
        row = [float(v) for v in p.pixels[y * p.width:(y + 1) * p.width]]
        if _std(row) < 5.0:
            continue
        m = _mean(row)
        r = [v - m for v in row]
        r0 = sum(v * v for v in r)
        ac = lambda lag: sum(a * b for a, b in zip(r, r[lag:])) / r0
        best, prev = 0.0, ac(1)
        for lag in range(2, 9):
            cur = ac(lag)
            if cur > prev and cur > 0:
                best = max(best, cur)
            prev = cur
        scores.append(best)
    return _mean(scores) if scores else None


def _aspect(e):
    w = _extent(e.contour)
    return _extent(e.contour, True) / w if w > 0 else 0.0


def glint_sigma(eye_width_px: float) -> float:
    return 0.05 + 3.0 / max(eye_width_px, 1.0)


def glint_llr(r: float, sigma: float) -> float:
    return min(max(((r - 1.0) ** 2 - (r - 0.5) ** 2) / (2 * sigma * sigma), -20.0), 20.0)


def score_frames(frames: List[FrameObs], calibration: Optional[Calibration] = None,
                 tier: str = "high_assurance", _non_monotonic: bool = False) -> ScanResult:
    cal = calibration or Calibration()
    n = len(frames)
    if _non_monotonic:
        return _early("fail", n, "TimestampNonMonotonic")
    if n < MIN_FRAMES:
        return _early("inconclusive", n, "TooFewFrames")
    if frames[-1].t_ms - frames[0].t_ms < MIN_SPAN_MS:
        return _early("inconclusive", n, "WindowTooShort")
    reasons: List[str] = []
    ev = dict(glint_llr=0.0, binocular_llr=0.0, glint_stability_llr=0.0, lip_llr=0.0,
              av_llr=0.0, blink_llr=0.0, jitter_llr=0.0, w_glint=0.0, w_lip=0.0, total_llr=0.0)
    t_s = [f.t_ms / 1000.0 for f in frames]
    dt_ms = max(_median([(b - a) * 1000 for a, b in zip(t_s, t_s[1:])]), 1.0)

    sigs = [_sig(f) for f in frames]
    if sum(1 for a, b in zip(sigs, sigs[1:]) if a == b) / (n - 1) > MAX_DUP_FRACTION:
        reasons.append("DuplicateFrames")

    meshes = [build_lip_mesh(f.lips_outer, f.lips_inner) for f in frames]
    mouth_w = _mean([m["width_px"] for m in meshes])
    open_ = [m["openness"] for m in meshes]
    spread = [m["spread"] for m in meshes]
    cents = [_centroid(m["vertices"]) for m in meshes]
    max_sd = 0.0
    mw = max(mouth_w, 1e-9)
    for k in range(len(meshes[0]["vertices"])):
        xs = [(m["vertices"][k][0] - c[0]) / mw for m, c in zip(meshes, cents)]
        ys = [(m["vertices"][k][1] - c[1]) / mw for m, c in zip(meshes, cents)]
        max_sd = max(max_sd, _std(xs), _std(ys))
    if max_sd < STATIC_MESH_EPS:
        reasons.append("StaticMesh")
    viseme = _articulation(open_, spread, t_s)
    if viseme < LIPSYNC_ACCEPT:
        reasons.append("WeakArticulation")
    wf = min(max((mouth_w - VISEME_CLIFF_PX) / (MOUTH_HIGH_ACC_PX - VISEME_CLIFF_PX), 0.0), 1.0)
    ev["lip_llr"] = 10.0 * (viseme - LIPSYNC_ACCEPT) * max(wf, 0.2)

    if all(f.audio_openness is not None for f in frames):
        audio = [f.audio_openness for f in frames]
        max_lag = math.ceil(400.0 / dt_ms)
        best_lag, best_c = 0, -math.inf
        for lag in range(-max_lag, max_lag + 1):
            c = _pearson_lag(open_, audio, lag)
            if c > best_c:
                best_c, best_lag = c, lag
        if abs(best_lag * dt_ms) > AV_MAX_LAG_MS or best_c < AV_MIN_CORR:
            reasons.append("AvDesync")
        ev["av_llr"] = min(max(8.0 * (_pearson_lag(open_, audio, 0) - 0.5), -8.0), 4.0)

    lock = [0.0, 0.0]; ratios = [None, None]; resid = [None, None]; light = [(0.0, 0.0)] * 2
    eye_ws = []; rect = dbl = locked_total = 0; any_motion = False
    for ei in range(2):
        di, dg, off = [], [], []
        for f in frames:
            e = f.left_eye if ei == 0 else f.right_eye
            w = max(_extent(e.contour), 1e-9)
            eye_ws.append(w)
            g, comps, area, fill = detect_glint(e)
            if g is None:
                continue
            locked_total += 1
            dbl += comps >= 2
            rect += area >= 16 and fill >= 0.95
            c = _centroid(e.contour)
            di.append(((e.iris_center[0] - c[0]) / w, (e.iris_center[1] - c[1]) / w))
            dg.append(((g[0] - c[0]) / w, (g[1] - c[1]) / w))
            off.append(((g[0] - e.iris_center[0]) / w, (g[1] - e.iris_center[1]) / w))
        lock[ei] = len(di) / n
        if off:
            light[ei] = _centroid(off)
        if lock[ei] < MIN_LOCK_FRACTION or len(di) < 3:
            continue
        mi, mg = _centroid(di), _centroid(dg)
        num = sum((a[0] - mi[0]) * (b[0] - mg[0]) + (a[1] - mi[1]) * (b[1] - mg[1]) for a, b in zip(di, dg))
        den = sum((a[0] - mi[0]) ** 2 + (a[1] - mi[1]) ** 2 for a in di)
        if math.sqrt(den / len(di)) < MIN_IRIS_MOTION:
            continue
        any_motion = True
        k = num / den
        ratios[ei] = abs(k)
        resid[ei] = math.sqrt(sum((mg[0] + k * (a[0] - mi[0]) - b[0]) ** 2 + (mg[1] + k * (a[1] - mi[1]) - b[1]) ** 2
                                  for a, b in zip(di, dg)) / len(di))
    eye_w = _mean(eye_ws)
    close = eye_w >= CLOSE_EYE_WIDTH_PX
    hint = "close" if close else "range"
    valid = [r for r in ratios if r is not None]
    arc = _mean(valid) if valid else 0.0
    sigma = glint_sigma(eye_w)
    if all(l < MIN_LOCK_FRACTION for l in lock):
        reasons.append("GlintNotLocked")
    elif not any_motion:
        reasons.append("NoOcularMotion")
        ev["glint_llr"] = -3.0 if close else -1.0
    else:
        ev["glint_llr"] = glint_llr(arc, sigma)
        if arc >= FLAT_GLINT_RATIO:
            reasons.append("FlatGlintGeometry")
        if len(valid) == 2:
            if abs(valid[0] - valid[1]) <= 3 * sigma:
                ev["binocular_llr"] = 1.0
            else:
                ev["binocular_llr"] = -4.0
                reasons.append("BinocularInconsistent")
        rs = [r for r in resid if r is not None]
        ev["glint_stability_llr"] = 1.5 if _mean(rs) < 0.02 else -3.0
    if locked_total:
        if dbl * 2 > locked_total:
            reasons.append("DoubledGlint")
        if rect * 2 > locked_total:
            reasons.append("RectangularGlint")
    (a0, a1), (b0, b1) = light
    m0, m1 = math.hypot(a0, a1), math.hypot(b0, b1)
    if m0 > 0.01 and m1 > 0.01 and (a0 * b0 + a1 * b1) / (m0 * m1) < LIGHT_MIN_COS:
        reasons.append("GlintLightMismatch")

    lm = [_landmarks(f) for f in frames]
    c0 = _centroid(lm[0])
    max_res = max(_affine_residual(lm[0], p) for p in lm[1:])
    moved = any(math.hypot(*(a - b for a, b in zip(_centroid(p), c0))) > 0.5 for p in lm)
    if moved and max_res / max(eye_w, 1e-9) < PLANAR_RESIDUAL_EPS:
        reasons.append("PlanarRigidMotion")

    asp = [0.5 * (_aspect(f.left_eye) + _aspect(f.right_eye)) for f in frames]
    med = _median(asp)
    if any(a < 0.6 * med for a in asp):
        ev["blink_llr"] = 1.5
    else:
        reasons.append("NoBlinkObserved")  # soft only: 5 s often has no blink
        ev["blink_llr"] = -0.5

    jit = []
    for k in range(2, n):
        a, b, c = frames[k - 2], frames[k - 1], frames[k]
        e2 = cnt = 0.0
        for ea, eb, ec in ((a.left_eye, b.left_eye, c.left_eye), (a.right_eye, b.right_eye, c.right_eye)):
            for p, q, r in zip(ea.contour, eb.contour, ec.contour):
                e2 += (p[0] - 2 * q[0] + r[0]) ** 2 + (p[1] - 2 * q[1] + r[1]) ** 2
                cnt += 1
        if cnt:
            jit.append(math.sqrt(e2 / cnt) / max(eye_w, 1e-9))
    # 2nd differences scale with dt^2: normalise to the 30 fps reference the range was set at,
    # so low-fps webcams/in-browser detectors (often 10-20 fps) are not flagged for normal motion.
    fps_norm = min(1.0, (REF_DT_MS / dt_ms) ** 2)
    jitter = _median(jit) * fps_norm
    static_face = "StaticMesh" in reasons and not any_motion
    if not static_face and not (JITTER_MIN <= jitter <= JITTER_MAX):
        reasons.append("JitterOutOfRange")
    elif not static_face:
        ev["jitter_llr"] = 1.0

    moire = [s for f in frames for s in (_moire(f.left_eye.patch), _moire(f.right_eye.patch)) if s is not None]
    if moire and sum(1 for s in moire if s > MOIRE_ROW_SCORE) * 2 > len(moire):
        reasons.append("MoirePattern")
    bright = []
    for f in frames:
        px = f.left_eye.patch.pixels + f.right_eye.patch.pixels
        bright.append(sum(px) / len(px) if px else 0.0)
    d = [b - a for a, b in zip(bright, bright[1:])]
    alts = sum(1 for x, y in zip(d, d[1:]) if abs(x) > FLICKER_MIN_STEP and abs(y) > FLICKER_MIN_STEP and _sign(x) != _sign(y))
    if len(d) > 2 and alts / (len(d) - 1) > FLICKER_ALT_FRACTION:
        reasons.append("ScreenFlicker")

    w_g, w_l = (1.0, 0.6) if close else (0.3, 1.0)
    ev["w_glint"], ev["w_lip"] = w_g, w_l
    ev["total_llr"] = (w_g * (ev["glint_llr"] + ev["binocular_llr"] + ev["glint_stability_llr"])
                       + w_l * (ev["lip_llr"] + ev["av_llr"]) + ev["blink_llr"] + ev["jitter_llr"])
    post = cal.posterior(ev["total_llr"])

    g = GlintReading(arc_ratio=arc, eye_width_px=eye_w, distance_hint=hint)
    l = LipsyncReading(mouth_width_px=mouth_w, viseme_match=viseme)
    legacy = fuse(g, l)
    vetoes = [r for r in reasons if r in VETOES]
    ocular_spoof = "NoOcularMotion" in reasons and close
    legacy_pass = legacy.passed and not vetoes and not ocular_spoof
    ha_pass = (not vetoes and not ocular_spoof and post >= HIGH_ASSURANCE_THRESHOLD
               and ev["total_llr"] >= llr_accept_threshold())
    if not legacy.passed:
        reasons.append("FusedBelowThreshold")
    if not ha_pass:
        reasons.append("BelowHighAssurance")
    passed = legacy_pass if tier == "legacy" else ha_pass
    if passed:
        reasons.append("Passed")
    return ScanResult(status="pass" if passed else "fail", tier=tier, fused_score=legacy.fused_score,
                      legacy_pass=legacy_pass, posterior=post, high_assurance_pass=ha_pass,
                      calibrated=cal.calibrated, evidence=ev, glint_score=g.score,
                      glint_arc_ratio=arc, lipsync_score=l.score, viseme_match=viseme,
                      distance_hint=hint, frames_used=n, glint_lock_fraction=tuple(lock),
                      vetoes=vetoes, reasons=reasons)


def _wipe(frames: List[FrameObs]) -> None:
    for f in frames:
        for e in (f.left_eye, f.right_eye):
            e.patch.pixels[:] = bytes(len(e.patch.pixels))
            e.contour[:] = [(0.0, 0.0)] * len(e.contour)
            e.iris_center = (0.0, 0.0)
        f.lips_outer[:] = [(0.0, 0.0)] * len(f.lips_outer)
        f.lips_inner[:] = [(0.0, 0.0)] * len(f.lips_inner)
        f.audio_openness = None
    frames.clear()


def scan_window(frames: List[FrameObs], *, consent: bool,
                calibration: Optional[Calibration] = None,
                tier: str = "high_assurance") -> ScanResult:
    """Score one window. Refuses to run without explicit consent. Frames past
    the 5 s window are ignored; the input list is wiped afterwards."""
    if not consent:
        return _early("inconclusive", 0, "NoConsent")
    kept, nonmono = [], False
    start = frames[0].t_ms if frames else 0
    for f in frames:
        if f.t_ms < start or f.t_ms - start > WINDOW_MS:
            break
        if kept and f.t_ms <= kept[-1].t_ms:
            nonmono = True
        kept.append(f)
    try:
        return score_frames(kept, calibration, tier, nonmono)
    finally:
        _wipe(frames)
        kept.clear()


# ---------- optional headless capture adapter ----------
# MediaPipe Face Mesh (refine_landmarks=True) indices.
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LEFT_IRIS, _RIGHT_IRIS = [468, 469, 470, 471, 472], [473, 474, 475, 476, 477]
_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def capture_and_scan(*, consent: bool, camera_index: int = 0,
                     calibration: Optional[Calibration] = None,
                     tier: str = "high_assurance") -> ScanResult:
    """Run the 5 s window headlessly from a webcam. No preview, overlay or
    drawing. Frames live only in memory and are wiped after scoring. Does not
    (and must not) suppress the OS/browser camera permission or indicator."""
    if not consent:
        return _early("inconclusive", 0, "NoConsent")
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional deps
        raise RuntimeError("capture_and_scan needs optional deps: pip install opencv-python mediapipe") from exc

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("camera unavailable or permission denied")
    frames: List[FrameObs] = []
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                           refine_landmarks=True)
    t0 = time.monotonic()
    try:
        while (now := time.monotonic()) - t0 <= WINDOW_MS / 1000.0:
            ok, bgr = cap.read()
            if not ok:
                continue
            h, w = bgr.shape[:2]
            res = mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            lmk = res.multi_face_landmarks[0].landmark
            P = lambda i: (lmk[i].x * w, lmk[i].y * h)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            def eye(contour_ids, iris_ids):
                cont = [P(i) for i in contour_ids]
                iris = [P(i) for i in iris_ids]
                ctr = iris[0]
                rad = _mean([math.hypot(p[0] - ctr[0], p[1] - ctr[1]) for p in iris[1:]])
                x0, x1 = int(min(p[0] for p in cont)) - 4, int(max(p[0] for p in cont)) + 4
                y0, y1 = int(min(p[1] for p in cont)) - 8, int(max(p[1] for p in cont)) + 8
                x0, y0 = max(x0, 0), max(y0, 0)
                crop = gray[y0:y1, x0:x1]
                return EyeObs(cont, ctr, rad, EyePatch((float(x0), float(y0)), crop.shape[1],
                                                       crop.shape[0], bytearray(crop.tobytes())))
            frames.append(FrameObs(int((now - t0) * 1000), eye(_LEFT_EYE, _LEFT_IRIS),
                                   eye(_RIGHT_EYE, _RIGHT_IRIS),
                                   [P(i) for i in _LIPS_OUTER], [P(i) for i in _LIPS_INNER]))
            del gray, bgr
    finally:
        cap.release()
        mesh.close()
    return scan_window(frames, consent=True, calibration=calibration, tier=tier)


# ---------- synthetic generator (tests / eval only; not a face model) ----------
LIVES = ["Live", "LiveRange", "LiveAudio"]
SPOOFS = ["PrintedStatic", "PrintedMoved", "ScreenFlatGlint", "ScreenMoire", "ScreenFlicker",
          "ScreenRectGlint", "ScreenDoubleGlint", "ScreenDuplicated", "DeepfakeDesync",
          "DeepfakeSmooth", "DeepfakeJittery", "DeepfakeLight"]


class _Rng:
    def __init__(self, s):
        self.s = s

    def f(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (self.s >> 33) / float(1 << 31) - 0.5


def _syn_eye(cx, cy, s, io, gr, light, blink, k, frame):
    w, h = int(90 * s), int(50 * s)
    ox, oy = cx - 45 * s, cy - 25 * s
    ix, iy = cx + io[0], cy + io[1]
    ir = 12.0 * s
    gx, gy = ix + light[0] * s + (gr - 1) * io[0], iy + light[1] * s + (gr - 1) * io[1]
    base = (124.0 if frame % 2 == 0 else 116.0) if k == "ScreenFlicker" else 120.0
    px = bytearray(w * h)
    for y in range(h):
        yy = y + oy
        for x in range(w):
            xx = x + ox
            v = base
            if k == "ScreenMoire":
                v += 25.0 * math.sin(xx * 2 * math.pi / 4.0)
            if (xx - ix) ** 2 + (yy - iy) ** 2 <= ir * ir:
                v = 40.0
            d2 = (xx - gx) ** 2 + (yy - gy) ** 2
            if k == "ScreenRectGlint":
                if abs(xx - gx) <= 2.5 and abs(yy - gy) <= 2.5:
                    v = 255.0
            elif k == "ScreenDoubleGlint":
                if d2 <= 2.25:
                    v = 255.0 - 40.0 * d2
                d3 = (xx - gx - 6.0) ** 2 + (yy - gy) ** 2
                if d3 <= 2.25:
                    v = 255.0 - 40.0 * d3
            elif d2 <= 2.25 * s * s:
                v = 255.0 - 40.0 * d2 / (s * s)
            px[y * w + x] = int(min(max(v, 0.0), 255.0))
    ry = (3.0 if blink else 12.0) * s
    cont = [(cx + 35 * s * math.cos(a), cy + ry * math.sin(a)) for a in (q / 16 * 2 * math.pi for q in range(16))]
    return EyeObs(cont, (ix, iy), ir, EyePatch((ox, oy), w, h, px))


def _syn_lips(cx, cy, s, op):
    ang = [q / 20 * 2 * math.pi for q in range(20)]
    o = [(cx + 60 * s * math.cos(a), cy + (14 + 20 * op) * s * math.sin(a)) for a in ang]
    i = [(cx + (45 - 4 * op) * s * math.cos(a), cy + (1 + 18 * op) * s * math.sin(a)) for a in ang]
    return o, i


def synth_sequence(k: str, n: int = 150, dt: int = 33, seed: int = 0) -> List[FrameObs]:
    r = _Rng(seed)
    phase = (seed % 7) * 0.37
    s = 0.6 if k == "LiveRange" else 1.0
    printed = k in ("PrintedStatic", "PrintedMoved")
    jit = 0.0 if k in ("DeepfakeSmooth", "PrintedStatic", "PrintedMoved") else (14.0 if k == "DeepfakeJittery" else 1.0)
    gr = 1.0 if k in ("ScreenFlatGlint", "PrintedStatic", "PrintedMoved") else 0.5
    open_f = lambda t: 0.5 + 0.3 * math.sin(2 * math.pi * 3 * t + phase) + 0.15 * math.sin(2 * math.pi * 1.13 * t)
    out = []
    for i in range(n):
        t = i * dt / 1000.0
        if k == "PrintedStatic":
            hx = hy = 0.0
        else:
            hx = 3 * math.sin(0.4 * t + phase) + jit * r.f()
            hy = 2 * math.cos(0.3 * t) + jit * r.f()
        io = (0.0, 0.0) if printed else (6 * s * math.sin(2 * t + phase), 2.5 * s * math.cos(1.3 * t))
        op = 0.3 if printed else open_f(t) + 0.03 * r.f()
        blink = (not printed) and k != "DeepfakeSmooth" and 2.0 <= t < 2.15
        le = (-2.0, -2.0) if k == "DeepfakeLight" else (2.0, -2.0)
        lo, li = _syn_lips(320 + hx, 400 + hy, 1.7 if k == "LiveRange" else 1.0, op)
        audio = open_f(t) if k == "LiveAudio" else (open_f(t - 0.25) if k == "DeepfakeDesync" else None)
        f = FrameObs(i * dt, _syn_eye(250 + hx, 250 + hy, s, io, gr, le, blink, k, i),
                     _syn_eye(250 + 140 * s + hx, 250 + hy, s, io, gr, (2.0, -2.0), blink, k, i),
                     lo, li, audio)
        if k == "PrintedMoved":
            ang, sc = 0.05 * math.sin(0.8 * t), 1 + 0.03 * math.sin(0.5 * t)
            c0, s0 = math.cos(ang) * sc, math.sin(ang) * sc
            tr = lambda p: (320 + c0 * (p[0] - 320) - s0 * (p[1] - 320), 320 + s0 * (p[0] - 320) + c0 * (p[1] - 320))
            for e in (f.left_eye, f.right_eye):
                e.contour = [tr(p) for p in e.contour]
                e.iris_center = tr(e.iris_center)
            f.lips_outer = [tr(p) for p in f.lips_outer]
            f.lips_inner = [tr(p) for p in f.lips_inner]
        out.append(f)
    if k == "ScreenDuplicated":
        import copy
        dup = []
        for i, f in enumerate(out):
            g = copy.deepcopy(out[i // 2]); g.t_ms = f.t_ms; dup.append(g)
        out = dup
    return out
