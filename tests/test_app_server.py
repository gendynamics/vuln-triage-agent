"""End-to-end checks of app/server.py handlers (no network)."""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import server as A


class TestServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        A.DATA = self.tmp
        A.RECORDS_PATH = os.path.join(self.tmp, "records.jsonl")
        A.RECORDS.clear()

    def run_kind(self, kind):
        code, out = A.do_run({"consent": True, "simulate": kind, "report": "SQL injection, auth: none", "tier": "nano"})
        self.assertEqual(code, 200)
        return out

    def test_consent_required(self):
        self.assertEqual(A.do_challenge({})[0], 403)
        self.assertEqual(A.do_scan({"simulate": "Live"})[0], 403)

    def test_live_passes_and_records(self):
        out = self.run_kind("Live")
        self.assertTrue(out["gate"]["aura_passed"])
        self.assertEqual(out["result"]["severity"], "critical")
        self.assertTrue(out["record"]["schema_valid"])
        self.assertEqual(out["record"]["challenge_match_ratio"], 1.0)
        text = open(A.RECORDS_PATH).read()
        for banned in ("lips_", "pixels", "contour", out["challenge"]["phrase"]):
            self.assertNotIn(banned, text)

    def test_spoof_blocked_with_vetoes(self):
        out = self.run_kind("ScreenMoire")
        self.assertTrue(out["record"]["blocked"])
        self.assertIn("MoirePattern", out["record"]["aura"]["vetoes"])

    def test_scan_id_single_use(self):
        _, g = A.do_scan({"consent": True, "simulate": "Live"})
        self.assertEqual(A.do_triage({"report": "x", "scan_id": g["scan_id"]})[0], 200)
        self.assertEqual(A.do_triage({"report": "x", "scan_id": g["scan_id"]})[0], 400)

    def test_demo_engine(self):
        self.assertEqual(A.demo_result("SSRF to 169.254.169.254")["attack_technique"], "T1552.005 - Cloud Instance Metadata API")


    def test_calibration_recording(self):
        """Opt-in calibration recording writes label + scalar evidence only; refuses without consent/label."""
        import json
        old = A.CAL_PATH
        A.CAL_PATH = os.path.join(tempfile.mkdtemp(), "calibration.jsonl")
        try:
            code, _ = A.do_scan({"consent": True, "simulate": "Live", "calibration_label": "live"})
            self.assertEqual(code, 400)  # no calibration_consent
            code, _ = A.do_scan({"consent": True, "simulate": "Live", "calibration_consent": True, "calibration_label": "cat"})
            self.assertEqual(code, 400)
            code, g = A.do_scan({"consent": True, "simulate": "ScreenMoire", "calibration_consent": True, "calibration_label": "screen_replay"})
            self.assertEqual(code, 200)
            rows = [json.loads(l) for l in open(A.CAL_PATH)]
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), {"label", "source", "evidence", "vetoes", "reasons", "frames_used"})
            self.assertEqual(rows[0]["label"], "screen_replay")
            self.assertNotIn("pixels", open(A.CAL_PATH).read())
        finally:
            A.CAL_PATH = old

    def test_webcam_shaped_frames(self):
        """Frames as sent by app/static/aura-camera.js (pixel coords + extra keys) are accepted and not persisted."""
        import aura_scan as S
        sim = S.synth_sequence("Live", seed=3)
        def eye(e):
            return {"contour": [list(p) for p in e.contour], "iris_center": list(e.iris_center), "iris_radius": e.iris_radius,
                    "patch": {"origin": list(e.patch.origin), "width": e.patch.width, "height": e.patch.height, "pixels": list(e.patch.pixels)}}
        frames = [{"t_ms": f.t_ms, "left_eye": eye(f.left_eye), "right_eye": eye(f.right_eye),
                   "lips_outer": [list(p) for p in f.lips_outer], "lips_inner": [list(p) for p in f.lips_inner],
                   "yaw": 3.0, "blend": {"jawOpen": 0.1}, "lip_z": [0, 0, 0, 0, 0, 0], "iod_px": 120.0} for f in sim]
        _, ch = A.do_challenge({"consent": True})
        body = {"consent": True, "frames": frames, "nonce": ch["nonce"]}
        code, g = A.do_scan(body)
        self.assertEqual(code, 200)
        self.assertEqual(g["source"], "camera")
        self.assertNotIn("frames", body)
        self.assertIsNotNone(g["challenge"])
        code, t = A.do_triage({"report": "x", "scan_id": g["scan_id"]})
        self.assertEqual(code, 200)
        text = open(A.RECORDS_PATH).read()
        self.assertNotIn("pixels", text); self.assertNotIn("contour", text)


if __name__ == "__main__":
    unittest.main()
