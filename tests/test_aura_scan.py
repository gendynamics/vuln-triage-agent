"""Synthetic tests for the AURA silent scan and challenge-response (Python mirror)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import aura_scan as S
import aura_challenge as C


class TestSilentScan(unittest.TestCase):
    def test_consent_required(self):
        r = S.scan_window(S.synth_sequence("Live", seed=1), consent=False)
        self.assertEqual(r.status, "inconclusive")
        self.assertIn("NoConsent", r.reasons)

    def test_live_passes_high_assurance(self):
        r = S.scan_window(S.synth_sequence("Live", seed=11), consent=True)
        self.assertTrue(r.passed, r.reasons)
        self.assertGreaterEqual(r.evidence["total_llr"], S.llr_accept_threshold())
        self.assertFalse(r.calibrated)

    def test_live_range_legacy_only(self):
        hi = S.scan_window(S.synth_sequence("LiveRange", seed=11), consent=True)
        lo = S.scan_window(S.synth_sequence("LiveRange", seed=11), consent=True, tier="legacy")
        self.assertFalse(hi.passed)
        self.assertTrue(lo.passed, lo.reasons)

    def test_every_spoof_vetoed(self):
        for k in S.SPOOFS:
            with self.subTest(k=k):
                r = S.scan_window(S.synth_sequence(k, seed=11), consent=True)
                self.assertFalse(r.passed)
                self.assertTrue(r.vetoes, r.reasons)
                lo = S.scan_window(S.synth_sequence(k, seed=11), consent=True, tier="legacy")
                self.assertFalse(lo.passed)

    def test_too_few_frames(self):
        r = S.scan_window(S.synth_sequence("Live", n=10, seed=1), consent=True)
        self.assertNotEqual(r.status, "pass")

    def test_window_and_wipe(self):
        frames = S.synth_sequence("Live", n=300, seed=11)  # ~10 s
        r = S.scan_window(frames, consent=True)
        self.assertLessEqual(r.frames_used, S.WINDOW_MS // 33 + 1)
        self.assertEqual(frames, [])
        d = r.to_dict()
        self.assertNotIn("pixels", str(d))

    def test_calibration_config(self):
        cal = S.Calibration.from_config("platt_a=0.5\nplatt_b=-1\ncalibrated=true\nisotonic=0:0,1:1")
        self.assertTrue(cal.calibrated)
        self.assertLess(cal.posterior(0.0), 0.5)


class TestChallenge(unittest.TestCase):
    def setUp(self):
        self.s = C.ChallengeSession(consent=True, seed=7)

    def test_requires_consent(self):
        with self.assertRaises(PermissionError):
            C.ChallengeSession(consent=False)

    def test_vocab_distinct_and_sizes(self):
        seqs = [tuple(C.word_visemes(i)) for i in range(len(C.VOCAB))]
        self.assertEqual(len(set(seqs)), len(seqs))
        ch = self.s.issue(0)
        self.assertTrue(C.MIN_WORDS <= len(ch.words) <= C.MAX_WORDS)

    def test_correct_phrase_passes(self):
        ch = self.s.issue(2000)
        o = self.s.verify(ch.nonce, C.synth_response(ch.words, 2400, seed=3))
        self.assertTrue(o.passed, o.codes)
        self.assertEqual(o.match_ratio, 1.0)
        self.assertNotIn("words", o.to_dict())

    def test_wrong_phrase_fails(self):
        ch = self.s.issue(2000)
        wrong = [w for w in range(len(C.VOCAB)) if w not in ch.words][:len(ch.words)]
        o = self.s.verify(ch.nonce, C.synth_response(wrong, 2400, seed=3))
        self.assertFalse(o.passed)
        self.assertIn("WrongPhrase", o.codes)

    def test_replay_fails(self):
        old = self.s.issue(2000)
        rec = C.synth_response(old.words, 2400, seed=3)
        self.assertTrue(self.s.verify(old.nonce, rec).passed)
        self.assertEqual(self.s.verify(old.nonce, rec).codes, ["NonceReplay"])
        new = self.s.issue(2000)
        self.assertFalse(self.s.verify(new.nonce, rec).passed)

    def test_early_start_fails(self):
        ch = self.s.issue(3000)
        o = self.s.verify(ch.nonce, C.synth_response(ch.words, 1500, seed=3))
        self.assertIn("EarlyStart", o.codes)

    def test_fusion_and_veto(self):
        ch = self.s.issue(1000)
        ok = self.s.verify(ch.nonce, C.synth_response(ch.words, 1500, seed=1))
        r = S.scan_window(S.synth_sequence("Live", seed=11), consent=True)
        before = r.evidence["total_llr"]
        C.apply_to_scan(r, ok)
        self.assertTrue(r.passed)
        self.assertGreater(r.evidence["total_llr"], before)
        ch2 = self.s.issue(1000)
        bad = self.s.verify(ch2.nonce, C.synth_response(ch2.words, 100, seed=1))
        r2 = C.apply_to_scan(S.scan_window(S.synth_sequence("Live", seed=11), consent=True), bad)
        self.assertFalse(r2.passed)
        self.assertIn("ChallengeFailed", r2.vetoes)

    def test_tts_fallback_and_speak_back_scoped(self):
        spoken = []
        fake = lambda t: (spoken.append(t), {"spoken": False, "engine": None, "display_text": t})[1]
        ch, res = C.prompt(self.s, now_ms=500, tts=fake)
        self.assertEqual(res["display_text"], "Please say: " + ch.phrase)
        self.assertEqual(ch.prompt_end_ms, 500)
        self.assertTrue(self.s.verify(ch.nonce, C.synth_response(ch.words, 900, seed=2)).passed)
        self.assertIsNotNone(self.s.speak_confirmation(tts=fake))
        self.assertIsNone(self.s.speak_confirmation(tts=fake))  # once only
        self.s.close()
        self.assertEqual(self.s.verify("x", []).codes, ["SessionClosed"])


if __name__ == "__main__":
    unittest.main()
