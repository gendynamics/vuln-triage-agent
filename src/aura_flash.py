"""Screen-flash colour challenge (active-light / photometric liveness) for AURA.

The server issues a random colour sequence bound to the challenge nonce. During capture the browser
shows it as a soft full-screen overlay. A live face lit by the screen reflects the colours with a
short camera latency; a pre-recorded or replayed video cannot, because it never saw this sequence.
The browser sends only mean RGB of skin ROIs (forehead, cheeks, the same ROIs as rPPG) plus one
background ROI per frame. No pixels and no images.

All thresholds are UNCALIBRATED design values.
"""
from __future__ import annotations

import math
import random
import secrets
from typing import Dict, List, Optional, Sequence, Tuple

PALETTE: Dict[str, Tuple[int, int, int]] = {  # moderate brightness, no pure white
    "red": (210, 40, 40), "green": (40, 190, 60), "blue": (50, 70, 220),
    "magenta": (200, 50, 190), "cyan": (40, 190, 200), "yellow": (210, 190, 40)}
SEG_MS = 450          # <= 2.2 colour transitions per second (photosensitivity limit is 3/s)
FADE_MS = 100
START_MS = 300        # after t0 (the same clock as frame t_ms)
N_SEG = 6
MAX_LAG_MS = 300
EXT_LAG_MS = 900
CORR_MATCH = 0.6
CORR_ALT = 0.8
LIGHT_MIN, LIGHT_MAX = 25.0, 235.0
MIN_RESP = 0.0015     # minimum chromaticity std for a "strong" response (FlashMismatch only)
SKIN = ("forehead", "lcheek", "rcheek")


def make_sequence(seed: Optional[int] = None, n: int = N_SEG) -> List[dict]:
    """n segments, no two consecutive colours equal, starting at START_MS."""
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    names, seq = list(PALETTE), []
    for i in range(n):
        c = rng.choice([x for x in names if not seq or x != seq[-1]["color"]])
        seq.append({"color": c, "rgb": list(PALETTE[c]), "start_ms": START_MS + i * SEG_MS, "dur_ms": SEG_MS})
    return seq


def _chroma(rgb):
    s = sum(rgb) or 1.0
    return [x / s for x in rgb]


def expected_at(seq: Sequence[dict], t: float) -> Optional[List[float]]:
    for s in seq:
        if s["start_ms"] <= t < s["start_ms"] + s["dur_ms"]:
            return _chroma(s["rgb"])
    return None


def _pearson(a, b):
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
    if va <= 1e-18 or vb <= 1e-18:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def _traces(extras: List[dict], key_set=SKIN):
    out = []
    for e in extras:
        r = e.get("rppg")
        if not isinstance(r, dict) or not all(k in r for k in key_set):
            continue
        rgb = [sum(float(r[k][c]) for k in key_set) / len(key_set) for c in range(3)]
        out.append((e["t_ms"], rgb))
    return out


def _corr_at_lag(tr, seq, lag):
    """Per-channel Pearson of the measured chromaticity vs the expected emitted chromaticity at t - lag."""
    xs, es = [], []
    for t, rgb in tr:
        e = expected_at(seq, t - lag)
        if e is not None:
            xs.append(_chroma(rgb)); es.append(e)
    if len(xs) < 6:
        return None
    return [_pearson([x[k] for x in xs], [y[k] for y in es]) for k in range(3)], len(xs)


def _best_lag(tr, seq, lo, hi, step=33):
    best = None
    for lag in range(lo, hi + 1, step):
        r = _corr_at_lag(tr, seq, lag)
        if r is None:
            continue
        c, n = r
        score = sum(sorted(c, reverse=True)[:2]) / 2
        if best is None or score > best[0]:
            best = (score, lag, c, n)
    return best


def flash_features(extras: List[dict], seq: Sequence[dict]) -> Optional[dict]:
    tr = _traces(extras)
    if not seq or len(tr) < 8:
        return None
    t0, t1 = seq[0]["start_ms"], seq[-1]["start_ms"] + seq[-1]["dur_ms"]
    win = [x for x in tr if t0 <= x[0] <= t1 + EXT_LAG_MS]
    luma = sum(0.299 * r + 0.587 * g + 0.114 * b for _, (r, g, b) in win) / max(len(win), 1)
    ch = [_chroma(rgb) for _, rgb in win]
    resp = max((sum((c[k] - sum(x[k] for x in ch) / len(ch)) ** 2 for c in ch) / len(ch)) ** 0.5 for k in range(3)) if ch else 0.0
    b = _best_lag(tr, seq, 0, MAX_LAG_MS)
    ext = _best_lag(tr, seq, MAX_LAG_MS + 33, EXT_LAG_MS)
    # lag consistency: best lag on each half of the sequence separately
    h = len(seq) // 2
    b1, b2 = _best_lag(tr, seq[:h], 0, EXT_LAG_MS), _best_lag(tr, seq[h:], 0, EXT_LAG_MS)
    # strong but wrong: best correlation against random alternative sequences on the same timing grid
    rng = random.Random(1234)
    alt = 0.0
    for _ in range(60):
        a = make_sequence(rng.randrange(1 << 30), len(seq))
        a = [{**s, "start_ms": o["start_ms"], "dur_ms": o["dur_ms"]} for s, o in zip(a, seq)]
        if [s["color"] for s in a] == [s["color"] for s in seq]:
            continue
        r = _best_lag(tr, a, 0, EXT_LAG_MS, step=66)
        if r:
            alt = max(alt, r[0])
    bg = _traces(extras, ("bg",))
    bgb = _best_lag(bg, seq, 0, MAX_LAG_MS) if len(bg) >= 8 else None
    return {"n": len(win), "luma": luma, "response": resp,
            "corr": b and [round(c, 3) for c in b[2]], "score": b and b[0], "lag_ms": b and b[1],
            "n_match": sum(1 for c in (b[2] if b else []) if c >= CORR_MATCH),
            "ext_score": ext and ext[0], "ext_lag_ms": ext and ext[1],
            "lag_halves_ms": [b1 and b1[1], b2 and b2[1]], "alt_score": alt,
            "bg_score": bgb and bgb[0], "skin_minus_bg": (b[0] - bgb[0]) if b and bgb else None}


def flash_channel(extras: List[dict], seq: Optional[Sequence[dict]]) -> Tuple[float, List[str], List[str], Optional[dict]]:
    """flash_llr:
      luma outside [25, 235] or < 8 frames            -> 0, "FlashInconclusive"
      >= 2 channels corr >= 0.6 at lag 0-300 ms, half-sequence lags within 150 ms -> +3
      match only at lag > 300 ms, or inconsistent lags -> -1, "ColorLagMismatch"
      strong response (chroma std >= 0.0015) that fits a different sequence (alt >= 0.8, own < 0.3)
                                                      -> -3, veto "FlashMismatch" (replayed old flash)
      otherwise (adequate light, no/weak response)    -> -3, "NoColorReflection"
    """
    if not seq:
        return 0.0, [], [], None
    f = flash_features(extras, seq)
    if f is None or f["n"] < 8 or f["score"] is None:
        return 0.0, ["FlashInconclusive"], [], f
    if not (LIGHT_MIN <= f["luma"] <= LIGHT_MAX):
        return 0.0, ["FlashInconclusive"], [], f
    l1, l2 = f["lag_halves_ms"]
    consistent = l1 is not None and l2 is not None and abs(l1 - l2) <= 150
    if f["n_match"] >= 2 and consistent:
        return 3.0, [], [], f
    if f["n_match"] >= 2 or (f["ext_score"] or 0) >= CORR_MATCH:
        return -1.0, ["ColorLagMismatch"], [], f
    if f["response"] >= MIN_RESP and f["alt_score"] >= CORR_ALT and f["score"] < 0.3:
        return -3.0, [], ["FlashMismatch"], f
    return -3.0, ["NoColorReflection"], [], f
