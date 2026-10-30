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


if __name__ == "__main__":
    unittest.main()
