import os, sys, unittest, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import aura_flash as F


def traces(seq, gain=0.25, lag=120, base=(150, 110, 95), noise=0.6, fps=30, scale=1.0, bg_gain=0.05):
    rng = random.Random(7); out = []
    for i in range(int(4500 * fps / 1000)):
        t = i * 1000 / fps
        cur = next((s for s in seq if s["start_ms"] <= t - lag < s["start_ms"] + s["dur_ms"]), None)
        def px(g):
            return [scale * (b + (g * (cur["rgb"][k] - 128) if cur else 0)) + rng.gauss(0, noise) for k, b in enumerate(base)]
        sk = px(gain)
        out.append({"t_ms": round(t), "rppg": {"forehead": sk, "lcheek": sk, "rcheek": sk, "bg": px(bg_gain)}})
    return out


class FlashTests(unittest.TestCase):
    def setUp(self):
        self.seq = F.make_sequence(5)

    def test_sequence_rules(self):
        for sd in range(50):
            s = F.make_sequence(sd)
            self.assertEqual(len(s), 6)
            self.assertTrue(all(a["color"] != b["color"] for a, b in zip(s, s[1:])))
            self.assertTrue(all(400 <= x["dur_ms"] <= 500 for x in s))
        self.assertLessEqual(1000 / F.SEG_MS, 3)

    def test_match(self):
        llr, rs, vs, f = F.flash_channel(traces(self.seq), self.seq)
        self.assertEqual((llr, rs, vs), (3.0, [], []), f)
        self.assertTrue(0 <= f["lag_ms"] <= 300)
        self.assertGreater(f["skin_minus_bg"], 0)

    def test_no_response(self):
        llr, rs, vs, _ = F.flash_channel(traces(self.seq, gain=0.0), self.seq)
        self.assertEqual((llr, rs), (-3.0, ["NoColorReflection"]))

    def test_wrong_sequence(self):
        other = F.make_sequence(99)
        self.assertNotEqual([s["color"] for s in other], [s["color"] for s in self.seq])
        llr, rs, vs, f = F.flash_channel(traces(other), self.seq)
        self.assertLess(llr, 0)
        self.assertTrue(vs == ["FlashMismatch"] or rs, f)

    def test_lag(self):
        llr, rs, vs, f = F.flash_channel(traces(self.seq, lag=600), self.seq)
        self.assertEqual((llr, rs), (-1.0, ["ColorLagMismatch"]), f)

    def test_low_light(self):
        llr, rs, vs, _ = F.flash_channel(traces(self.seq, scale=0.1), self.seq)
        self.assertEqual((llr, rs), (0.0, ["FlashInconclusive"]))

    def test_absent(self):
        self.assertEqual(F.flash_channel(traces(self.seq), None)[0], 0.0)


if __name__ == "__main__":
    unittest.main()


class FlashServerTests(unittest.TestCase):
    def test_challenge_has_flash_and_scan_reports_it(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
        import server
        code, ch = server.do_challenge({"consent": True})
        self.assertEqual(code, 200)
        self.assertEqual(len(ch["flash"]["seq"]), 6)
        self.assertEqual(ch["flash"]["fade_ms"], 100)
