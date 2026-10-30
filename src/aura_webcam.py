"""Webcam-only evidence channels for AURA (Python only; the Rust crate keeps the canonical
scan math). Inputs are the per-frame extras sent by app/static/aura-camera.js: pixel-space lip
and iris points, MediaPipe blendshapes, head pose (yaw/pitch from the facial transformation
matrix), canonical lip z-depths. Nothing here is stored; callers clear the lists afterwards.

Every channel returns (llr, reasons) and is folded into a ScanResult with apply_channel(),
which re-derives posterior / tiers exactly like aura_challenge.apply_to_scan.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

Pt = Tuple[float, float]

# positions inside aura_scan._LIPS_OUTER / _LIPS_INNER
_O = {0: 15, 37: 16, 267: 14, 17: 5, 84: 4, 314: 6, 61: 0, 291: 10}
_I = {13: 15, 14: 5, 78: 0, 308: 10}


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _std(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _pearson(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = _mean(a), _mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den > 1e-12 else 0.0


def smooth3(v: Sequence[float]) -> List[float]:
    """Light [1,2,1]/4 smoothing (edges kept)."""
    if len(v) < 3:
        return list(v)
    return [v[0]] + [(v[i - 1] + 2 * v[i] + v[i + 1]) / 4 for i in range(1, len(v) - 1)] + [v[-1]]


def deltas(v: Sequence[float]) -> Tuple[List[float], List[float]]:
    s = smooth3(v)
    d1 = [b - a for a, b in zip(s, s[1:])]
    d2 = [b - a for a, b in zip(d1, d1[1:])]
    return d1, d2


# ---------- (1) geometry: Procrustes-normalised lips ----------
def procrustes(points: Sequence[Pt], iris_l: Pt, iris_r: Pt) -> List[Pt]:
    """Translate to the inter-ocular midpoint, de-roll so the eye line is horizontal,
    scale by the inter-ocular distance (IOD). Removes in-plane pose and camera distance."""
    cx, cy = (iris_l[0] + iris_r[0]) / 2, (iris_l[1] + iris_r[1]) / 2
    dx, dy = iris_r[0] - iris_l[0], iris_r[1] - iris_l[1]
    iod = math.hypot(dx, dy) or 1.0
    c, s = dx / iod, dy / iod  # rotate by -roll
    return [(((x - cx) * c + (y - cy) * s) / iod, (-(x - cx) * s + (y - cy) * c) / iod) for x, y in points]


def lip_geometry(outer: Sequence[Pt], inner: Sequence[Pt]) -> Dict[str, float]:
    """MAR = mean of 3 vertical outer-lip distances / mouth width; inner h/w."""
    d = lambda p, q: math.hypot(p[0] - q[0], p[1] - q[1])
    w = d(outer[_O[61]], outer[_O[291]]) or 1e-9
    mar = (d(outer[_O[0]], outer[_O[17]]) + d(outer[_O[37]], outer[_O[84]]) + d(outer[_O[267]], outer[_O[314]])) / 3 / w
    iw = d(inner[_I[78]], inner[_I[308]]) or 1e-9
    return {"mar": mar, "inner_hw": d(inner[_I[13]], inner[_I[14]]) / iw, "width_iod": w}


def geometry_features(extras: List[dict]) -> Dict[str, list]:
    mar, ihw, width, protr, yaw, jaw, pucker = [], [], [], [], [], [], []
    for f in extras:
        o = procrustes(f["lips_outer"], f["iris_l"], f["iris_r"])
        i = procrustes(f["lips_inner"], f["iris_l"], f["iris_r"])
        g = lip_geometry(o, i)
        mar.append(g["mar"]); ihw.append(g["inner_hw"]); width.append(g["width_iod"])
        z = f.get("lip_z")
        iod = f.get("iod_px") or 1.0
        # protrusion proxy: lip centre depth relative to the mouth corners (canonical mesh z), / IOD
        protr.append(((z[0] + z[1]) / 2 - (z[4] + z[5]) / 2) / iod if z and len(z) >= 6 else 0.0)
        yaw.append(f.get("yaw") or 0.0)
        b = f.get("blend") or {}
        jaw.append(b.get("jawOpen", 0.0)); pucker.append(max(b.get("mouthPucker", 0.0), b.get("mouthFunnel", 0.0)))
    k = min(5, len(width))
    base_w = sorted(width[:k])[k // 2] if k else 1.0
    base_p = sorted(protr[:k])[k // 2] if k else 0.0
    rounding = [base_w / w - 1.0 if w > 0 else 0.0 for w in width]   # narrower than baseline → rounded
    protrusion = [base_p - p for p in protr]                         # more negative z = toward camera
    return {"mar": mar, "inner_hw": ihw, "rounding": rounding, "protrusion": protrusion,
            "yaw": yaw, "jaw": jaw, "pucker": pucker}


def geometry_channel(extras: List[dict]) -> Tuple[float, List[str]]:
    """Articulation evidence from pose-normalised geometry:
    - speech-like dynamics: std of Δ MAR in a natural band (static photo ~0, codec noise tiny)
    - cross-estimator consistency: Procrustes MAR vs the blendshape jawOpen (two independent
      estimators of the same jaw motion should agree on a real face)
    - rounding/protrusion activity, weighted ×1.5 when 20° <= |yaw| <= 45° (depth is observable)."""
    if len(extras) < 10:
        return 0.0, ["GeometryInsufficient"]
    g = geometry_features(extras)
    d1, _ = deltas(g["mar"])
    motion = _std(d1)
    reasons, llr = [], 0.0
    if motion < 0.004:
        reasons.append("GeometryStatic"); llr -= 2.0
    elif motion <= 0.15:
        llr += 1.5
    else:
        reasons.append("GeometryErratic"); llr -= 1.0
    if any(j > 0 for j in g["jaw"]) and _std(g["mar"]) > 1e-3:
        c = _pearson(g["mar"], g["jaw"])
        if c >= 0.5:
            llr += 1.5
        elif c < 0.1:
            reasons.append("GeometryBlendshapeMismatch"); llr -= 1.5
    side = _mean([1.0 if 20 <= abs(y) <= 45 else 0.0 for y in g["yaw"]]) > 0.3
    rp = _std(g["rounding"]) + _std(g["protrusion"])
    if rp > 0.01:
        llr += 0.5 * (1.5 if side else 1.0)
    return llr, reasons


def deroll_lips(f: dict) -> Tuple[List[Pt], List[Pt]]:
    """Lips rotated so the eye line is horizontal (pixel scale kept) for the challenge features."""
    o = procrustes(f["lips_outer"], f["iris_l"], f["iris_r"])
    i = procrustes(f["lips_inner"], f["iris_l"], f["iris_r"])
    return o, i


# ---------- shared ----------
def apply_channel(r, name: str, llr: float, reasons: Sequence[str] = (), vetoes: Sequence[str] = (), calibration=None):
    """Fold a webcam channel's LLR into a ScanResult (mutates and returns it)."""
    from aura_scan import Calibration, HIGH_ASSURANCE_THRESHOLD, llr_accept_threshold
    if r.status == "inconclusive":
        return r
    cal = calibration or Calibration()
    r.evidence[name] = llr
    r.evidence["total_llr"] += llr
    r.posterior = cal.posterior(r.evidence["total_llr"])
    for c in reasons:
        if c not in r.reasons:
            r.reasons.append(c)
    for v in vetoes:
        if v not in r.vetoes:
            r.vetoes.append(v)
            if v not in r.reasons:
                r.reasons.append(v)
    if vetoes:
        r.legacy_pass = False
    ocular = "NoOcularMotion" in r.reasons and r.distance_hint == "close"
    r.high_assurance_pass = (not r.vetoes and not ocular and r.posterior >= HIGH_ASSURANCE_THRESHOLD
                             and r.evidence["total_llr"] >= llr_accept_threshold())
    r.reasons = [c for c in r.reasons if c not in ("Passed", "BelowHighAssurance")]
    if not r.high_assurance_pass:
        r.reasons.append("BelowHighAssurance")
    ok = r.legacy_pass if r.tier == "legacy" else r.high_assurance_pass
    if ok:
        r.reasons.append("Passed")
    r.status = "pass" if ok else "fail"
    return r


def extras_from_json(arr: List[dict]) -> List[dict]:
    out = []
    for f in arr:
        try:
            out.append({"t_ms": int(f["t_ms"]), "lips_outer": [tuple(map(float, p)) for p in f["lips_outer"]],
                        "lips_inner": [tuple(map(float, p)) for p in f["lips_inner"]],
                        "iris_l": tuple(map(float, f["left_eye"]["iris_center"])),
                        "iris_r": tuple(map(float, f["right_eye"]["iris_center"])),
                        "yaw": f.get("yaw"), "pitch": f.get("pitch"), "blend": f.get("blend"),
                        "lip_z": f.get("lip_z"), "iod_px": f.get("iod_px"),
                        "mesh3d": f.get("mesh3d"), "mesh2d": f.get("mesh2d"), "rppg": f.get("rppg")})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def apply_webcam_channels(r, scan_extras: List[dict], all_extras: Optional[List[dict]] = None):
    """Run every webcam channel on the scan window and fold it in."""
    llr, rs = geometry_channel(scan_extras)
    apply_channel(r, "geometry_llr", llr, rs)
    return r
