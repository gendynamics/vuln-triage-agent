"""triage() accepts an AURA silent-scan result and blocks when the gate fails."""
import json, os, sys, unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agent import triage, scan_gate
import aura_scan as S

GOOD = {"severity": "high", "cvss_score": 7.5, "attack_technique": "T1190", "summary": "x",
        "remediation_steps": ["a"], "affected_assets": ["b"], "estimated_effort": "S", "priority": 2}


class TestScanHook(unittest.TestCase):
    def test_failed_scan_blocks_without_model_call(self):
        r = S.scan_window(S.synth_sequence("ScreenMoire", seed=11), consent=True)
        with patch("agent._client") as c:
            out = triage("report", tier="nano", aura_scan=r)
            c.assert_not_called()
        self.assertIsNone(out["triage"])
        self.assertIn("MoirePattern", out["aura_gate"]["vetoes"])

    def test_passed_scan_dict_allows_triage(self):
        d = S.scan_window(S.synth_sequence("Live", seed=11), consent=True).to_dict()
        self.assertIsNone(scan_gate(d))
        with patch("agent._client") as c:
            c.return_value.chat.completions.create.return_value.choices = [
                MagicMock(message=MagicMock(content=json.dumps(GOOD)))]
            out = triage("report", tier="nano", aura_scan=d)
        self.assertEqual(out["severity"], "high")
        self.assertTrue(out["aura_scan"]["aura_passed"])

    def test_no_scan_is_noop(self):
        self.assertIsNone(scan_gate(None))


if __name__ == "__main__":
    unittest.main()
