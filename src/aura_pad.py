"""AURA presentation-attack-detection (PAD) channel: pretrained MiniFASNetV2 face anti-spoofing.

Model: MiniFASNetV2 (2.7_80x80) from minivision-ai/Silent-Face-Anti-Spoofing (Apache-2.0), ONNX export
garciafido/minifasnet-v2-anti-spoofing-onnx (Apache-2.0, 1.7 MB). It is downloaded at runtime into a
cache dir and SHA-256 checked; it is not vendored. Upstream does not document its training data, so we
make no claim about it, and its score is UNCALIBRATED.

Input: up to MAX_CROPS face crops (80x80, BGR, 0..255, 2.7x the face box), sampled across the scan by
the browser. Crops are only scored in memory and then zeroed. Nothing is stored, and there is no identity matching.
Output: pad_llr in [-PAD_LLR_CAP, +PAD_LLR_CAP] plus reason codes:
  PADInconclusive  no crops / model unavailable -> llr 0
  PADSpoof         every crop confidently spoof (p_live < VETO_P, >= 3 crops) -> veto
"""
from __future__ import annotations

import hashlib, math, os, urllib.request
from typing import List, Optional, Sequence, Tuple

MODEL_URL = "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx"
MODEL_SHA256 = "d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b"
CROP = 80
MAX_CROPS = 5
LIVE_INDEX = 1          # upstream Silent-Face test.py: argmax == 1 -> real face
PAD_LLR_CAP = 3.0
LLR_SCALE = 0.75        # shrink the raw logit: model is uncalibrated on webcam data
VETO_P = 0.05
VETO_MIN_CROPS = 3

_session = None
_load_error: Optional[str] = None


def _cache_path() -> str:
    d = os.environ.get("AURA_PAD_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "vulntriage")
    return os.path.join(d, "minifasnet_v2.onnx")


def load_model(download: bool = True):
    """Return an onnxruntime session or None (never raises). Set AURA_PAD_DISABLE=1 to switch off."""
    global _session, _load_error
    if _session is not None:
        return _session
    if os.environ.get("AURA_PAD_DISABLE") == "1":
        _load_error = "disabled"
        return None
    try:
        import onnxruntime as ort
        path = os.environ.get("AURA_PAD_MODEL") or _cache_path()
        if not os.path.exists(path):
            if not download:
                raise FileNotFoundError(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".part"
            urllib.request.urlretrieve(MODEL_URL, tmp)
            if hashlib.sha256(open(tmp, "rb").read()).hexdigest() != MODEL_SHA256:
                os.remove(tmp)
                raise ValueError("model checksum mismatch")
            os.replace(tmp, path)
        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _load_error = None
    except Exception as e:  # missing onnxruntime, offline, bad file: channel goes inconclusive
        _load_error = f"{type(e).__name__}: {e}"
        _session = None
    return _session


def _softmax(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
    return [v / s for v in e]


def score_crops(crops: Sequence[Sequence[float]], session=None) -> List[float]:
    """p_live per crop. Each crop is a flat HWC BGR list of 80*80*3 values in 0..255 (raw, not /255:
    the upstream pipeline feeds unnormalised pixels)."""
    import numpy as np
    sess = session or load_model()
    if sess is None:
        raise RuntimeError(_load_error or "PAD model unavailable")
    ok = [c for c in crops if len(c) == CROP * CROP * 3][:MAX_CROPS]
    if not ok:
        return []
    x = np.asarray(ok, dtype=np.float32).reshape(-1, CROP, CROP, 3).transpose(0, 3, 1, 2).copy()
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    x.fill(0)
    return [_softmax([float(v) for v in row])[LIVE_INDEX] for row in out]


def fuse_pad(p_live: Sequence[float]) -> Tuple[float, List[str], List[str], dict]:
    """Pure fusion math (model-free). Mean p_live -> scaled logit, capped at +-PAD_LLR_CAP."""
    if not p_live:
        return 0.0, ["PADInconclusive"], [], {"n": 0}
    p = min(max(sum(p_live) / len(p_live), 0.02), 0.98)
    llr = max(-PAD_LLR_CAP, min(PAD_LLR_CAP, LLR_SCALE * math.log(p / (1 - p))))
    reasons, vetoes = [], []
    if len(p_live) >= VETO_MIN_CROPS and max(p_live) < VETO_P:
        vetoes.append("PADSpoof")
    info = {"n": len(p_live), "p_live_mean": round(sum(p_live) / len(p_live), 4),
            "p_live": [round(v, 4) for v in p_live], "llr": round(llr, 3), "calibrated": False}
    return llr, reasons, vetoes, info


def pad_channel(crops, session=None) -> Tuple[float, List[str], List[str], dict]:
    """(llr, reasons, vetoes, info). Never raises; model failure -> PADInconclusive, llr 0."""
    if not crops:
        return 0.0, ["PADInconclusive"], [], {"n": 0, "error": "no face crops"}
    try:
        p = score_crops(crops, session)
    except Exception as e:
        return 0.0, ["PADInconclusive"], [], {"n": 0, "error": str(e)[:200]}
    return fuse_pad(p)
