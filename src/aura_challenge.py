#!/usr/bin/env python3
"""AURA lip-reading CHALLENGE-RESPONSE — mirror of crates/aura-rs/src/challenge.rs.

A random 4-6 word phrase from a fixed vocabulary, bound to a one-time nonce,
is spoken (TTS) or displayed; the person repeats it during the scan and we
verify the lip-mesh motion matches that phrase, after the prompt ended.

Closed-vocabulary verification of a *prompted* phrase only — this is NOT,
and must not be extended into, open-ended transcription of arbitrary speech.
Only a match score and pass/fail leave `verify()`; decoded words are never
returned or stored. Thresholds are heuristic and uncalibrated.

Plug a real visual speech recognizer (Auto-AVSR, AV-HuBERT; not bundled) in by
passing any object with `decode(feats) -> [(word_or_None, start_ms, end_ms)]`
restricted to VOCAB.
"""

from __future__ import annotations

import math
import secrets
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Protocol, Sequence, Tuple

VISEME_TARGETS = {  # (openness, spread)
    "Rest": (0.03, 0.72), "P": (0.00, 0.55), "F": (0.06, 0.88), "A": (0.30, 0.75),
    "O": (0.18, 0.45), "E": (0.10, 0.95), "T": (0.13, 0.72),
}
VISEMES = ["Rest", "P", "F", "A", "O", "E", "T"]

_P2V = {}
for _v, _ps in {"P": "P B M", "F": "F V", "A": "AA AE AH AY AW",
                "O": "OW UW W OY R ER UH AO", "E": "IY IH EH EY S Z CH SH JH ZH"}.items():
    for _p in _ps.split():
        _P2V[_p] = _v


def phoneme_to_viseme(p: str) -> str:
    return _P2V.get(p, "T")


VOCAB: List[Tuple[str, str]] = [
    ("blue", "B L UW"), ("five", "F AY V"), ("seven", "S EH V AH N"), ("tiger", "T AY G ER"),
    ("moon", "M UW N"), ("apple", "AE P AH L"), ("river", "R IH V ER"), ("green", "G R IY N"),
    ("map", "M AE P"), ("window", "W IH N D OW"), ("pizza", "P IY T S AH"), ("violet", "V AY AH L AH T"),
    ("coffee", "K AA F IY"), ("ocean", "OW SH AH N"),
]

MIN_WORDS, MAX_WORDS = 4, 6
CHALLENGE_WINDOW_MS = 8000
MAX_WORD_COST = 0.08
MIN_MATCH_RATIO = 0.75
MIN_MS_PER_WORD = 150.0


def word_visemes(i: int) -> List[str]:
    out: List[str] = []
    for p in VOCAB[i][1].split():
        v = phoneme_to_viseme(p)
        if not out or out[-1] != v:
            out.append(v)
    return out


def _vd(a, b):
    return math.hypot(3 * (a[0] - b[0]), a[1] - b[1])


@dataclass
class LipFeat:
    t_ms: int
    openness: float
    spread: float


@dataclass
class Challenge:
    nonce: str
    words: List[int]
    prompt_end_ms: int

    @property
    def phrase(self) -> str:
        return " ".join(VOCAB[i][0] for i in self.words)


class VisualSpeechRecognizer(Protocol):
    def decode(self, feats: Sequence[LipFeat]) -> List[Tuple[Optional[int], int, int]]: ...


def snap(f: LipFeat) -> str:
    return min(VISEMES, key=lambda v: _vd(VISEME_TARGETS[v], (f.openness, f.spread)))


def _dtw(a: List[str], b: List[str]) -> float:
    n, m = len(a), len(b)
    if not n or not m:
        return math.inf
    d = [[math.inf] * (m + 1) for _ in range(n + 1)]
    d[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = _vd(VISEME_TARGETS[a[i - 1]], VISEME_TARGETS[b[j - 1]])
            d[i][j] = c + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[n][m] / max(n, m)


class MeshDtwRecognizer:
    """Heuristic baseline: viseme run-length segmentation + per-word DTW."""

    def decode(self, feats):
        runs = []  # [viseme, count, t0, t1]
        for f in feats:
            v = snap(f)
            if runs and runs[-1][0] == v:
                runs[-1][1] += 1
                runs[-1][3] = f.t_ms
            else:
                runs.append([v, 1, f.t_ms, f.t_ms])
        runs = [r for r in runs if r[1] >= 2]
        out, cur = [], []

        def flush():
            if not cur:
                return
            seq = []
            for c in cur:
                if not seq or seq[-1] != c[0]:
                    seq.append(c[0])
            costs = [(_dtw(seq, word_visemes(w)), w) for w in range(len(VOCAB))]
            bc, bw = min(costs)
            out.append((bw if bc <= MAX_WORD_COST else None, cur[0][1], cur[-1][2]))
            cur.clear()

        for v, k, t0, t1 in runs:
            if v == "Rest":
                if k >= 3:
                    flush()
            else:
                cur.append((v, t0, t1))
        flush()
        return out


@dataclass
class ChallengeOutcome:
    match_ratio: float
    llr: float
    onset_after_prompt_ms: int
    passed: bool
    codes: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _fail(*codes):
    return ChallengeOutcome(0.0, -10.0, 0, False, list(codes))


class ChallengeSession:
    """Issues challenges; remembers spent nonces in memory for this session only.
    Requires consent. Nothing is written to disk."""

    def __init__(self, *, consent: bool, seed: Optional[int] = None):
        if not consent:
            raise PermissionError("challenge-response requires explicit consent")
        import random
        self._rng = random.Random(seed) if seed is not None else secrets.SystemRandom()
        self._active: Optional[Challenge] = None
        self._spent: set = set()
        self._confirm_words: Optional[List[int]] = None
        self.open = True

    def issue(self, prompt_end_ms: int = 0) -> Challenge:
        n = self._rng.randint(MIN_WORDS, MAX_WORDS)
        words = self._rng.sample(range(len(VOCAB)), n)
        nonce = "%032x" % self._rng.getrandbits(128)
        while nonce in self._spent:
            nonce = "%032x" % self._rng.getrandbits(128)
        self._active = Challenge(nonce, words, prompt_end_ms)
        return self._active

    def verify(self, nonce: str, feats: Sequence[LipFeat],
               vsr: Optional[VisualSpeechRecognizer] = None) -> ChallengeOutcome:
        if not self.open:
            return _fail("SessionClosed")
        if nonce in self._spent:
            return _fail("NonceReplay")
        ch = self._active
        if ch is None or ch.nonce != nonce:
            return _fail("UnknownNonce")
        self._active = None
        self._spent.add(nonce)
        feats = [f for f in feats if f.t_ms <= ch.prompt_end_ms + CHALLENGE_WINDOW_MS]
        spans = (vsr or MeshDtwRecognizer()).decode(feats)
        if not spans:
            return _fail("NoSpeech")
        got = [s[0] for s in spans]
        exp = ch.words
        n, m = len(exp), len(got)
        d = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            d[i][0] = i
        for j in range(m + 1):
            d[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sub = 0 if got[j - 1] == exp[i - 1] else 1
                d[i][j] = min(d[i - 1][j - 1] + sub, d[i - 1][j] + 1, d[i][j - 1] + 1)
        i, j, matched = n, m, 0
        while i > 0 and j > 0:
            sub = 0 if got[j - 1] == exp[i - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + sub:
                matched += 1 - sub
                i, j = i - 1, j - 1
            elif d[i][j] == d[i - 1][j] + 1:
                i -= 1
            else:
                j -= 1
        ratio = matched / n
        onset = spans[0][1] - ch.prompt_end_ms
        dur = spans[-1][2] - spans[0][1]
        codes = []
        if onset < 0:
            codes.append("EarlyStart")
        if dur / n < MIN_MS_PER_WORD:
            codes.append("TooFast")
        if ratio < MIN_MATCH_RATIO:
            codes.append("WrongPhrase")
        passed = not codes
        if passed:
            codes.append("Matched")
            self._confirm_words = list(ch.words)  # in memory, this session only
        got.clear()
        return ChallengeOutcome(ratio, max(-10.0, min(6.0, 12 * (ratio - 0.5))), onset, passed, codes)

    def speak_confirmation(self, tts=None) -> Optional[dict]:
        """Speak the verified phrase back once, within this consented session."""
        if not self.open or self._confirm_words is None:
            return None
        text = "Verified: " + " ".join(VOCAB[i][0] for i in self._confirm_words)
        self._confirm_words = None
        return (tts or speak)(text)

    def close(self):
        self.open = False
        self._active = None
        self._confirm_words = None
        self._spent.clear()


def speak(text: str) -> dict:
    """TTS with graceful fallback: pyttsx3 -> macOS `say` -> `espeak` -> display text.
    Returns {"spoken": bool, "engine": str|None, "display_text": str}."""
    try:
        import pyttsx3  # optional
        eng = pyttsx3.init()
        eng.say(text)
        eng.runAndWait()
        return {"spoken": True, "engine": "pyttsx3", "display_text": text}
    except Exception:
        pass
    for exe in ("say", "espeak"):
        if shutil.which(exe):
            try:
                subprocess.run([exe, text], check=True, timeout=30)
                return {"spoken": True, "engine": exe, "display_text": text}
            except Exception:
                pass
    return {"spoken": False, "engine": None, "display_text": text}


def prompt(session: ChallengeSession, now_ms: int, tts=None, est_ms_per_word: int = 450) -> Tuple[Challenge, dict]:
    """Issue a challenge, speak/display it, and set prompt_end_ms (scan clock)."""
    ch = session.issue(0)
    res = (tts or speak)("Please say: " + ch.phrase)
    ch.prompt_end_ms = now_ms + (0 if not res.get("spoken") else est_ms_per_word * (len(ch.words) + 2))
    return ch, res


def apply_to_scan(r, o: ChallengeOutcome, calibration=None):
    """Fold the challenge LLR into an aura_scan.ScanResult (mutates and returns it)."""
    from aura_scan import Calibration, HIGH_ASSURANCE_THRESHOLD, llr_accept_threshold
    if r.status == "inconclusive":
        return r
    cal = calibration or Calibration()
    r.evidence["challenge_llr"] = o.llr
    r.evidence["total_llr"] += o.llr
    r.posterior = cal.posterior(r.evidence["total_llr"])
    if not o.passed:
        r.vetoes.append("ChallengeFailed")
        r.reasons.append("ChallengeFailed")
    r.legacy_pass = r.legacy_pass and o.passed
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


def synth_response(words: Sequence[int], start_ms: int, frames_per_viseme: int = 4, seed: int = 0) -> List[LipFeat]:
    import random
    rng = random.Random(seed)
    out, t = [], start_ms

    def push(v, k):
        nonlocal t
        o, s = VISEME_TARGETS[v]
        for _ in range(k):
            out.append(LipFeat(t, max(0.0, o + rng.uniform(-0.005, 0.005)), s + rng.uniform(-0.005, 0.005)))
            t += 33

    for w in words:
        for v in word_visemes(w):
            push(v, frames_per_viseme)
        push("Rest", 5)
    return out
