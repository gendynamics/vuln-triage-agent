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


# ---------------- attention / gaze (Face ID "attention aware" analogue) ----------------
ATT_MIN_FRAMES = 10


def attention_features(extras: List[dict]) -> Optional[dict]:
    """Per frame: horizontal gaze h = mean over eyes |lookIn - lookOut|, vertical v = |lookUp - lookDown|,
    eyes open = mean(eyeBlinkL/R) < 0.5, head |yaw| <= 30 and |pitch| <= 25 degrees.
    attentive frame = open AND h < 0.5 AND v < 0.5 AND head pose ok."""
    rows = []
    for e in extras:
        b = e.get("blend") or {}
        if "eyeLookInLeft" not in b:
            continue
        g = lambda k: float(b.get(k, 0.0))
        h = (abs(g("eyeLookInLeft") - g("eyeLookOutLeft")) + abs(g("eyeLookInRight") - g("eyeLookOutRight"))) / 2
        v = (abs(g("eyeLookUpLeft") - g("eyeLookDownLeft")) + abs(g("eyeLookUpRight") - g("eyeLookDownRight"))) / 2
        closed = (g("eyeBlinkLeft") + g("eyeBlinkRight")) / 2 >= 0.5
        pose_ok = abs(e.get("yaw") or 0.0) <= 30 and abs(e.get("pitch") or 0.0) <= 25
        rows.append((h, v, closed, pose_ok))
    if len(rows) < ATT_MIN_FRAMES:
        return None
    n = len(rows)
    att = sum(1 for h, v, c, p in rows if not c and h < 0.5 and v < 0.5 and p) / n
    return {"attentive_frac": att, "closed_frac": sum(1 for r in rows if r[2]) / n, "n": n}


def attention_channel(extras: List[dict]) -> Tuple[float, List[str]]:
    """LLR: attentive >= 60% of frames +0.5; < 30% -1.5 with soft-fail reason "Inattentive"
    (also "EyesClosed" when eyes are shut in > 50% of frames). No data -> 0."""
    f = attention_features(extras)
    if f is None:
        return 0.0, []
    rs = []
    if f["closed_frac"] > 0.5:
        rs.append("EyesClosed")
    if f["attentive_frac"] >= 0.6:
        return 0.5, rs
    if f["attentive_frac"] < 0.3:
        return -1.5, rs + ["Inattentive"]
    return 0.0, rs


# ---------------- temporal frame consistency ----------------
TELEPORT_IOD = 0.5     # eye-midpoint jump between consecutive frames, in inter-ocular distances
TELEPORT_MAX_DT = 100  # ms; longer gaps are "FrameGap", not teleports
GAP_MS = 250


def consistency_features(extras: List[dict]) -> Optional[dict]:
    """teleports: frames whose eye midpoint moves > 0.5 IOD within <= 100 ms (face swap / splice / cut).
    roughness: mean|d2 x| / mean|dx| of jawOpen and smile blendshapes (smooth motion < 1, white noise ~1.5-2).
    timing: coefficient of variation of frame intervals and count of gaps > 250 ms."""
    if len(extras) < 10:
        return None
    tele = gaps = 0
    for a, b in zip(extras, extras[1:]):
        dt = b["t_ms"] - a["t_ms"]
        if dt > GAP_MS:
            gaps += 1
            continue
        iod = math.dist(b["iris_l"], b["iris_r"]) or 1.0
        ma = ((a["iris_l"][0] + a["iris_r"][0]) / 2, (a["iris_l"][1] + a["iris_r"][1]) / 2)
        mb = ((b["iris_l"][0] + b["iris_r"][0]) / 2, (b["iris_l"][1] + b["iris_r"][1]) / 2)
        if dt <= TELEPORT_MAX_DT and math.dist(ma, mb) / iod > TELEPORT_IOD:
            tele += 1
    rough = []
    for k in ("jawOpen", "mouthSmileLeft", "mouthSmileRight"):
        v = [float((e.get("blend") or {}).get(k, float("nan"))) for e in extras]
        if any(math.isnan(x) for x in v):
            continue
        d1, d2 = [b - a for a, b in zip(v, v[1:])], []
        d2 = [b - a for a, b in zip(d1, d1[1:])]
        m1 = sum(map(abs, d1)) / len(d1)
        if m1 > 0.004:  # only when the channel actually moves
            rough.append(sum(map(abs, d2)) / len(d2) / m1)
    dts = [b["t_ms"] - a["t_ms"] for a, b in zip(extras, extras[1:]) if b["t_ms"] > a["t_ms"]]
    cv = _std(dts) / _mean(dts) if dts and _mean(dts) > 0 else 0.0
    return {"teleports": tele, "gaps": gaps, "roughness": max(rough) if rough else None, "dt_cv": cv}


def consistency_channel(extras: List[dict]) -> Tuple[float, List[str]]:
    """LLR: teleports >= 2 -> -2 "FrameTeleport" (1 -> -0.5); roughness > 1.4 -> -1 "BlendshapeJitter",
    < 1.0 -> +0.5; gaps or dt CV > 0.5 -> -0.5 "IrregularTiming"."""
    f = consistency_features(extras)
    if f is None:
        return 0.0, []
    llr, rs = 0.0, []
    if f["teleports"] >= 2:
        llr -= 2; rs.append("FrameTeleport")
    elif f["teleports"] == 1:
        llr -= 0.5
    if f["roughness"] is not None:
        if f["roughness"] > 1.4:
            llr -= 1; rs.append("BlendshapeJitter")
        elif f["roughness"] < 1.0:
            llr += 0.5
    if f["gaps"] or f["dt_cv"] > 0.5:
        llr -= 0.5; rs.append("IrregularTiming")
    return llr, rs


# ---------------- planar vs 3D (parallax) ----------------
PLANAR_RES = 0.012   # median homography residual / IOD below this with enough pose change -> flat
DEPTH_RES = 0.02
MIN_YAW_RANGE = 8.0  # degrees of head rotation needed before the test means anything


def _solve(A, b):
    n = len(b); M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit_homography(src: Sequence[Pt], dst: Sequence[Pt]):
    """Least-squares DLT with h33 = 1 (normal equations). Coordinates normalised for conditioning."""
    def norm(pts):
        cx = _mean([p[0] for p in pts]); cy = _mean([p[1] for p in pts])
        s = _mean([math.hypot(p[0] - cx, p[1] - cy) for p in pts]) or 1.0
        return [((p[0] - cx) / s, (p[1] - cy) / s) for p in pts], (cx, cy, s)
    a, ta = norm(src); b, tb = norm(dst)
    rows, rhs = [], []
    for (x, y), (u, v) in zip(a, b):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y]); rhs.append(v)
    AtA = [[sum(r[i] * r[j] for r in rows) for j in range(8)] for i in range(8)]
    Atb = [sum(r[i] * y for r, y in zip(rows, rhs)) for i in range(8)]
    h = _solve(AtA, Atb)
    if h is None:
        return None
    def apply(p):
        x, y = (p[0] - ta[0]) / ta[2], (p[1] - ta[1]) / ta[2]
        w = h[6] * x + h[7] * y + 1
        return ((h[0] * x + h[1] * y + h[2]) / w * tb[2] + tb[0], (h[3] * x + h[4] * y + h[5]) / w * tb[2] + tb[1])
    return apply


def planarity_features(extras: List[dict]) -> Optional[dict]:
    """For each frame vs the most-frontal frame: fit a homography to the 12 sparse landmarks and take the
    median residual / IOD. A flat photo/screen maps exactly by a homography; a 3D face shows parallax
    (nose vs cheeks) once the head rotates. Report the 90th-percentile residual over frames + yaw range."""
    fr = [e for e in extras if e.get("mesh2d") and len(e["mesh2d"]) >= 8 and e.get("yaw") is not None]
    if len(fr) < 10:
        return None
    yaws = [e["yaw"] for e in fr]
    ref = min(fr, key=lambda e: abs(e["yaw"]))
    src = [tuple(p) for p in ref["mesh2d"]]
    res = []
    for e in fr:
        dst = [tuple(p) for p in e["mesh2d"]]
        H = fit_homography(src, dst)
        if H is None:
            continue
        iod = math.dist(e["iris_l"], e["iris_r"]) or 1.0
        d = sorted(math.dist(H(s_), t) for s_, t in zip(src, dst))
        res.append(d[len(d) // 2] / iod)
    if not res:
        return None
    res.sort()
    return {"residual_p90": res[min(len(res) - 1, int(0.9 * len(res)))], "yaw_range": max(yaws) - min(yaws), "n": len(res)}


def planarity_channel(extras: List[dict]) -> Tuple[float, List[str], List[str]]:
    """Needs >= 8 deg yaw range (else 0, "PoseTooStatic3D"). residual < 0.012 IOD -> veto "PlanarFace";
    > 0.02 -> +1.0 (3D parallax present)."""
    f = planarity_features(extras)
    if f is None:
        return 0.0, [], []
    if f["yaw_range"] < MIN_YAW_RANGE:
        return 0.0, ["PoseTooStatic3D"], []
    if f["residual_p90"] < PLANAR_RES:
        return -3.0, [], ["PlanarFace"]
    if f["residual_p90"] > DEPTH_RES:
        return 1.0, [], []
    return 0.0, [], []


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


def apply_webcam_channels(r, scan_extras: List[dict], all_extras: Optional[List[dict]] = None, calibration=None):
    """Run every webcam channel on the scan window and fold it in."""
    llr, rs = geometry_channel(scan_extras)
    apply_channel(r, "geometry_llr", llr, rs, calibration=calibration)
    llr, rs = attention_channel(scan_extras)
    apply_channel(r, "attention_llr", llr, rs, calibration=calibration)
    llr, rs = consistency_channel(scan_extras)
    apply_channel(r, "consistency_llr", llr, rs, calibration=calibration)
    llr, rs, vs = planarity_channel(scan_extras)
    apply_channel(r, "planarity_llr", llr, rs, vs, calibration=calibration)
    return r
