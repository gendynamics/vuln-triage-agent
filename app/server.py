#!/usr/bin/env python3
"""VulnTriage + AURA bundled app — zero-dependency (stdlib) backend.

    python app/server.py            # http://localhost:8000

GET  /                    dashboard (app/static/index.html)
GET  /api/health          backend + mode info
POST /api/aura/challenge  {consent, tts?}            -> {nonce, phrase, prompt_end_ms, tts}
POST /api/aura/scan       {consent, simulate|frames, nonce?, challenge_response?} -> scan result
POST /api/triage          {report, tier, mode, scan_id?} -> triage result (+ record)
POST /api/run             {report, tier, mode, simulate, consent} -> gate+challenge+triage in one call
GET  /api/records         VulnTriage records (in memory; mirrored to app/data/records.jsonl)

Records hold scores, pass tiers, reason codes, challenge match ratio and the
triage JSON only: never images, landmarks, or recognized words. Simulated
landmark sequences are synthetic and labeled as such.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))

import aura_scan as S  # noqa: E402
import aura_challenge as C  # noqa: E402

try:
    import agent  # needs `openai` (requirements.txt); only required for live mode
except Exception as e:  # pragma: no cover
    agent = None
    AGENT_ERR = str(e)
else:
    AGENT_ERR = None

STATIC = os.path.join(ROOT, "static")
DATA = os.path.join(ROOT, "data")
RECORDS_PATH = os.path.join(DATA, "records.jsonl")
LOCK = threading.Lock()
RECORDS: list = []
PENDING: dict = {}  # scan_id -> gate summary (in memory, consumed by /api/triage)
# One challenge session per server process (demo scope; spent nonces kept in memory only).
SESSION = C.ChallengeSession(consent=True)

# ---------------- demo keyword engine (port of the dashboard's demoResult) ----------------
RULES = [
    ("rce", r"remote code|\brce\b|command (execution|injection)|deserializ|shell command|os\.system|eval\(", 9.8,
     "T1190 - Exploit Public-Facing Application", "L", 1,
     ["Disable or sandbox the vulnerable component/delegate immediately.", "Patch the library to a fixed version and pin it.",
      "Run the service as an unprivileged user in an isolated container with no ambient credentials.",
      "Rotate any secrets reachable from the host environment.", "Hunt for indicators of prior exploitation in host and egress logs."]),
    ("auth", r"auth(entication)? bypass|alg=none|missing (jwt|signature)|idor|broken access|privilege escalation", 9.1,
     "T1078 - Valid Accounts", "M", 1,
     ["Enforce signature verification and reject alg=none tokens.", "Add server-side authorization checks on every admin route.",
      "Invalidate all active sessions and rotate signing keys.", "Add regression tests for forged and tampered tokens.",
      "Review admin audit logs for unauthorized actions."]),
    ("sqli", r"sql|injection|or 1=1|union select", 9.8, "T1190 - Exploit Public-Facing Application", "M", 1,
     ["Replace string-built SQL with parameterized queries / prepared statements.",
      "Validate and allow-list the input parameter; enforce length limits.", "Stop reflecting database errors in responses.",
      "Run the service under a least-privilege DB role.", "Rotate session tokens and review DB audit logs for prior exploitation.",
      "Add SAST/DAST SQLi checks in CI."]),
    ("ssrf", r"ssrf|server-side request|169\.254\.169\.254|metadata", 8.6, "T1552.005 - Cloud Instance Metadata API", "M", 2,
     ["Allow-list outbound destinations and block link-local / private ranges.", "Enforce IMDSv2 (session tokens) on all instances.",
      "Route preview fetches through an egress proxy without cloud credentials.", "Rotate instance IAM credentials that may have been exposed."]),
    ("traversal", r"path traversal|directory traversal|\.\./|etc/passwd|lfi", 7.5, "T1083 - File and Directory Discovery", "S", 2,
     ["Resolve and canonicalize paths; reject anything outside the export directory.", "Serve downloads by opaque ID instead of file name.",
      "Move secrets out of readable config files and rotate DB passwords."]),
    ("xss", r"xss|cross-site scripting|<script|onerror=", 6.1, "T1059.007 - JavaScript", "S", 3,
     ["Context-aware output encoding for the rendered field.", "Sanitize stored HTML with an allow-list sanitizer.",
      "Add a strict Content-Security-Policy and HttpOnly session cookies."]),
    ("csrf", r"csrf|cross-site request", 5.4, "T1185 - Browser Session Hijacking", "S", 3,
     ["Add anti-CSRF tokens.", "Set SameSite=Lax/Strict on session cookies."]),
    ("info", r"version disclosure|banner|verbose error|stack trace|information disclosure", 3.7,
     "T1592 - Gather Victim Host Information", "S", 4, ["Suppress version banners and verbose errors in production."]),
]


def demo_result(report: str) -> dict:
    t = report or ""
    rule = next((r for r in RULES if re.search(r[1], t, re.I)), None)
    m = re.search(r"^\s*title:\s*(.+)$", t, re.I | re.M)
    title = m.group(1).strip() if m else None
    assets = []
    for x in re.findall(r"(?:GET|POST|PUT|DELETE|PATCH)?\s*(/[A-Za-z0-9_\-/{}.*]+)", t):
        x = x.strip()
        if len(x) > 2 and not x.startswith("//") and x not in assets:
            assets.append(x)
    assets = assets[:3]
    m = re.search(r"(?:Database|Component):\s*([^\n]+)", t, re.I)
    if m and m.group(1).strip() not in assets:
        assets.append(m.group(1).strip())
    unauth = bool(re.search(r"auth:\s*none|unauthenticated", t, re.I))
    if not rule:
        return {"severity": "informational", "cvss_score": 2.0 if t.strip() else 0.0, "attack_technique": "T1595 - Active Scanning",
                "summary": "Demo heuristic could not match a known vulnerability class" + (f' in "{title}"' if title else "") + "; manual review recommended.",
                "remediation_steps": ["Reproduce and confirm the issue.", "Add details: endpoint, parameter, PoC and impact."],
                "affected_assets": assets or ["unspecified"], "estimated_effort": "S", "priority": 5}
    k, _, cv, tech, eff, pri, steps = rule
    if not unauth and cv >= 9 and k != "auth":
        cv = round((cv - 0.9) * 10) / 10
    sev = "critical" if cv >= 9 else "high" if cv >= 7 else "medium" if cv >= 4 else "low" if cv > 0 else "informational"
    return {"severity": sev, "cvss_score": cv, "attack_technique": tech,
            "summary": (title or "Reported issue") + f" — classified by the demo keyword engine as {k.upper()}" + (" (no authentication required)" if unauth else "") + ".",
            "remediation_steps": list(steps), "affected_assets": assets or ["unspecified"], "estimated_effort": eff,
            "priority": 1 if sev == "critical" else pri}


# ---------------- AURA helpers ----------------
def _pt(p):
    return (float(p[0]), float(p[1]))


def frames_from_json(arr) -> list:
    """Parse browser/upstream-detector frames. Eye patch pixels: list of 0-255 ints."""
    out = []
    for f in arr:
        def eye(e):
            pa = e["patch"]
            return S.EyeObs(contour=[_pt(p) for p in e["contour"]], iris_center=_pt(e["iris_center"]),
                            iris_radius=float(e["iris_radius"]),
                            patch=S.EyePatch(origin=_pt(pa["origin"]), width=int(pa["width"]), height=int(pa["height"]),
                                             pixels=bytearray(int(v) & 255 for v in pa["pixels"])))
        out.append(S.FrameObs(t_ms=int(f["t_ms"]), left_eye=eye(f["left_eye"]), right_eye=eye(f["right_eye"]),
                              lips_outer=[_pt(p) for p in f["lips_outer"]], lips_inner=[_pt(p) for p in f["lips_inner"]],
                              audio_openness=f.get("audio_openness")))
    return out


def simulated_response(kind: str, ch: C.Challenge) -> list:
    """Synthetic lip features answering the challenge — simulation only."""
    if kind in S.LIVES:
        return C.synth_response(ch.words, ch.prompt_end_ms + 400, seed=1)
    if kind.startswith("Deepfake"):  # pre-rendered: starts before the prompt ends
        return C.synth_response(ch.words, max(0, ch.prompt_end_ms - 1000), seed=1)
    wrong = [w for w in range(len(C.VOCAB)) if w not in ch.words][:len(ch.words)]
    return C.synth_response(wrong, ch.prompt_end_ms + 400, seed=1)  # replayed/static media: wrong phrase


def gate_summary(r: S.ScanResult, ch_out, simulated: str | None) -> dict:
    d = r.to_dict()
    return {"status": r.status, "aura_passed": r.passed, "tier": r.tier, "posterior": r.posterior,
            "evidence": d["evidence"], "legacy_score": r.fused_score, "legacy_pass": r.legacy_pass,
            "high_assurance_pass": r.high_assurance_pass, "calibrated": r.calibrated,
            "vetoes": list(r.vetoes), "reasons": list(r.reasons), "distance_hint": r.distance_hint,
            "frames_used": r.frames_used, "glint_score": r.glint_score, "lipsync_score": r.lipsync_score,
            "fused_score": r.fused_score, "glint_arc_ratio": r.glint_arc_ratio, "viseme_match": r.viseme_match,
            "challenge": None if ch_out is None else {"match_ratio": ch_out.match_ratio, "passed": ch_out.passed,
                                                       "codes": list(ch_out.codes), "llr": ch_out.llr},
            "simulated": simulated}


def do_challenge(body):
    if body.get("consent") is not True:
        return 403, {"error": "consent=true required"}
    ch = SESSION.issue(0)
    ch.prompt_end_ms = 450 * (len(ch.words) + 2)  # estimated end of spoken/displayed prompt (scan clock)
    tts = C.speak("Please say: " + ch.phrase) if body.get("tts") else {"spoken": False, "engine": None,
                                                                        "display_text": "Please say: " + ch.phrase}
    return 200, {"nonce": ch.nonce, "phrase": ch.phrase, "prompt_end_ms": ch.prompt_end_ms, "tts": tts}


def do_scan(body):
    if body.get("consent") is not True:
        return 403, {"error": "consent=true required"}
    kind = body.get("simulate")
    if kind:
        if kind not in S.LIVES + S.SPOOFS:
            return 400, {"error": "unknown simulate kind", "kinds": S.LIVES + S.SPOOFS}
        frames = S.synth_sequence(kind, seed=int(body.get("seed", 11)))
    elif isinstance(body.get("frames"), list):
        try:
            frames = frames_from_json(body["frames"])
        except Exception as e:
            return 400, {"error": f"bad frames: {e}"}
    else:
        return 400, {"error": "provide simulate=<kind> or frames=[...]"}
    body.pop("frames", None)  # raw webcam frames (eye crops) live only in this call
    tier = body.get("tier_gate", "high_assurance")
    nonce = body.get("nonce")
    ch = SESSION._active if nonce and SESSION._active and SESSION._active.nonce == nonce else None
    camera_feats = None
    if not kind:
        # Webcam: challenge lip features come from the same frames (computed server-side);
        # the silent scan scores the 5 s window starting when the prompt ends (user speaking).
        camera_feats = []
        for f in frames:
            m = S.build_lip_mesh(f.lips_outer, f.lips_inner)
            camera_feats.append(C.LipFeat(f.t_ms, m["openness"], m["spread"]))
        if ch and not body.get("challenge_response"):
            later = [f for f in frames if f.t_ms >= ch.prompt_end_ms]
            if len(later) >= S.MIN_FRAMES:
                frames = later
    r = S.scan_window(frames, consent=True, tier=tier)  # zeroes pixels after scoring
    frames = None
    ch_out = None
    if nonce:
        if kind and ch:
            feats = simulated_response(kind, ch)
        elif body.get("challenge_response"):
            feats = [C.LipFeat(int(x["t_ms"]), float(x["openness"]), float(x["spread"]))
                     for x in body.get("challenge_response", [])]
        else:
            feats = camera_feats or []
        ch_out = SESSION.verify(nonce, feats)
        feats.clear()
        C.apply_to_scan(r, ch_out)
    g = gate_summary(r, ch_out, kind)
    g["source"] = "simulated" if kind else "camera"
    if camera_feats is not None:
        camera_feats.clear()
    sid = uuid.uuid4().hex
    with LOCK:
        PENDING[sid] = g
    return 200, {"scan_id": sid, **g}


def do_triage(body):
    report = body.get("report", "")
    tier = body.get("tier", "ultra")
    mode = body.get("mode", "demo")
    gate = None
    if body.get("scan_id"):
        with LOCK:
            gate = PENDING.pop(body["scan_id"], None)
        if gate is None:
            return 400, {"error": "unknown or already-used scan_id"}
    used = "demo"
    if gate is not None and not gate["aura_passed"]:
        result = {"aura_gate": gate, "triage": None, "error": "AURA liveness scan failed"}
    elif mode == "live" and os.environ.get("NEBIUS_API_KEY") and agent is not None:
        try:
            result = agent.triage(report, tier=tier, aura_scan=gate)
            used = "live"
        except BaseException as e:  # SystemExit from _client on a bad key, network errors
            result = demo_result(report)
            result["live_error"] = str(e)[:200]
    else:
        result = demo_result(report)
    valid = True
    if result.get("triage", 1) is not None:
        try:
            (agent._validate if agent else (lambda x: x))(result)
        except Exception as e:
            valid, result["validation_error"] = False, str(e)
    rec = make_record(gate, result, tier, used, valid)
    return 200, {"result": result, "record": rec, "engine": used}


def make_record(gate, result, tier, engine, valid):
    rec = {"id": uuid.uuid4().hex[:12], "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "tier": tier, "engine": engine, "schema_valid": valid,
           "blocked": result.get("triage", 1) is None,
           "aura": None if gate is None else {k: gate[k] for k in (
               "posterior", "evidence", "legacy_score", "legacy_pass", "high_assurance_pass", "aura_passed",
               "vetoes", "reasons", "calibrated", "simulated", "frames_used")},
           "challenge_match_ratio": None if not gate or not gate.get("challenge") else gate["challenge"]["match_ratio"],
           "triage": {k: v for k, v in result.items() if k != "aura_gate"}}
    with LOCK:
        RECORDS.append(rec)
        os.makedirs(DATA, exist_ok=True)
        with open(RECORDS_PATH, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return rec


def do_run(body):
    code, ch = do_challenge(body)
    if code != 200:
        return code, ch
    code, g = do_scan({**body, "nonce": ch["nonce"]})
    if code != 200:
        return code, g
    code, t = do_triage({**body, "scan_id": g["scan_id"]})
    return code, {"challenge": {"phrase": ch["phrase"], "tts": ch["tts"]}, "gate": g, **t}


def load_records():
    if os.path.exists(RECORDS_PATH):
        with open(RECORDS_PATH) as fh:
            for line in fh:
                try:
                    RECORDS.append(json.loads(line))
                except ValueError:
                    pass


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj, ctype="application/json"):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *a):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % a))

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as fh:
                return self._send(200, fh.read(), "text/html; charset=utf-8")
        if p == "/aura-camera.js":
            with open(os.path.join(STATIC, "aura-camera.js"), "rb") as fh:
                return self._send(200, fh.read(), "text/javascript; charset=utf-8")
        if p == "/api/health":
            return self._send(200, {"ok": True, "live_available": bool(os.environ.get("NEBIUS_API_KEY")) and agent is not None,
                                    "agent_error": AGENT_ERR, "simulate_kinds": {"live": S.LIVES, "spoof": S.SPOOFS}})
        if p == "/api/records":
            with LOCK:
                return self._send(200, {"records": list(RECORDS)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        routes = {"/api/aura/challenge": do_challenge, "/api/aura/scan": do_scan, "/api/triage": do_triage, "/api/run": do_run}
        fn = routes.get(self.path.split("?")[0])
        if not fn:
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 20_000_000:
                return self._send(413, {"error": "too large"})
            body = json.loads(self.rfile.read(n) or b"{}")
            code, obj = fn(body)
        except ValueError as e:
            code, obj = 400, {"error": f"bad json: {e}"}
        self._send(code, obj)


def main():
    load_records()
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"VulnTriage + AURA on http://localhost:{port}  (live mode: {'on' if os.environ.get('NEBIUS_API_KEY') else 'off — demo engine'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
