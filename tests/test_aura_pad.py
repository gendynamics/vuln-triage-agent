"""AURA PAD channel: fusion math with a mocked model, graceful failure, optional real-model smoke test."""
import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import aura_pad as P
import aura_scan as S
import aura_webcam as W
sys.path.insert(0, os.path.dirname(__file__))


class _In:
    name = "input"


class FakeSession:
    """Returns fixed logits per crop; class index 1 = live."""
    def __init__(self, p_live):
        self.p = p_live
    def get_inputs(self):
        return [_In()]
    def run(self, _o, feeds):
        n = feeds["input"].shape[0]
        import numpy as np
        rows = [[math.log((1 - self.p) / 2 + 1e-9), math.log(self.p + 1e-9), math.log((1 - self.p) / 2 + 1e-9)] for _ in range(n)]
        return [np.asarray(rows, dtype="float32")]


CROP = [128] * (80 * 80 * 3)


class PadFusionTests(unittest.TestCase):
    def test_live_positive_and_capped(self):
        llr, rs, vs, info = P.fuse_pad([0.999] * 5)
        self.assertGreater(llr, 0); self.assertLessEqual(llr, P.PAD_LLR_CAP)
        self.assertEqual(vs, []); self.assertFalse(info["calibrated"])

    def test_spoof_negative_capped_and_veto(self):
        llr, rs, vs, _ = P.fuse_pad([0.001] * 5)
        self.assertLess(llr, 0); self.assertGreaterEqual(llr, -P.PAD_LLR_CAP)
        self.assertIn("PADSpoof", vs)

    def test_no_veto_with_few_crops_or_one_live(self):
        self.assertEqual(P.fuse_pad([0.01, 0.01])[2], [])
        self.assertEqual(P.fuse_pad([0.01, 0.01, 0.6])[2], [])

    def test_neutral_is_zero(self):
        self.assertAlmostEqual(P.fuse_pad([0.5])[0], 0.0, places=6)

    def test_empty_inconclusive(self):
        llr, rs, vs, _ = P.fuse_pad([])
        self.assertEqual((llr, rs, vs), (0.0, ["PADInconclusive"], []))


class PadChannelTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")

    def test_mocked_model_scores(self):
        llr, rs, vs, info = P.pad_channel([CROP] * 3, session=FakeSession(0.9))
        self.assertEqual(info["n"], 3); self.assertAlmostEqual(info["p_live_mean"], 0.9, places=3)
        self.assertAlmostEqual(llr, P.LLR_SCALE * math.log(9), places=3)

    def test_only_max_crops_scored(self):
        self.assertEqual(P.pad_channel([CROP] * 9, session=FakeSession(0.9))[3]["n"], P.MAX_CROPS)

    def test_bad_crop_sizes_ignored(self):
        self.assertIn("PADInconclusive", P.pad_channel([[1, 2, 3]], session=FakeSession(0.9))[1])

    def test_model_failure_inconclusive(self):
        class Boom(FakeSession):
            def run(self, *a):
                raise RuntimeError("boom")
        llr, rs, vs, info = P.pad_channel([CROP] * 3, session=Boom(0.5))
        self.assertEqual((llr, rs, vs), (0.0, ["PADInconclusive"], []))

    def test_no_crops_inconclusive(self):
        self.assertEqual(P.pad_channel([])[1], ["PADInconclusive"])


class PadFusionIntoScanTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")

    def _scan(self):
        return S.scan_window(S.synth_sequence("Live", seed=11), consent=True)

    def test_pad_llr_added_to_total(self):
        r = self._scan(); before = r.evidence["total_llr"]
        llr, rs, vs, _ = P.pad_channel([CROP] * 3, session=FakeSession(0.9))
        W.apply_channel(r, "pad_llr", llr, rs, vs)
        self.assertAlmostEqual(r.evidence["pad_llr"], llr)
        self.assertAlmostEqual(r.evidence["total_llr"], before + llr)

    def test_pad_spoof_vetoes(self):
        r = self._scan()
        llr, rs, vs, _ = P.pad_channel([CROP] * 4, session=FakeSession(0.001))
        W.apply_channel(r, "pad_llr", llr, rs, vs)
        self.assertIn("PADSpoof", r.vetoes); self.assertFalse(r.high_assurance_pass)

    def test_apply_webcam_channels_with_pad_session(self):
        from test_aura_webcam import extras
        r = self._scan()
        W.apply_webcam_channels(r, extras(), pad_crops=[CROP] * 3, pad_session=FakeSession(0.8))
        self.assertIn("pad_llr", r.evidence); self.assertEqual(r.pad_info["n"], 3)

    def test_unavailable_model_keeps_rest(self):
        from test_aura_webcam import extras
        old = os.environ.get("AURA_PAD_DISABLE"); os.environ["AURA_PAD_DISABLE"] = "1"
        saved, P._session = P._session, None
        try:
            r = self._scan()
            W.apply_webcam_channels(r, extras(), pad_crops=[CROP] * 3)
            self.assertEqual(r.evidence["pad_llr"], 0.0)
            self.assertIn("PADInconclusive", r.reasons); self.assertIn("geometry_llr", r.evidence)
        finally:
            P._session = saved
            if old is None: os.environ.pop("AURA_PAD_DISABLE", None)
            else: os.environ["AURA_PAD_DISABLE"] = old


class PadRealModelSmoke(unittest.TestCase):
    """Runs only if onnxruntime is installed and the model is already cached (no network in tests)."""
    def test_real_model_runs(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")
        path = os.environ.get("AURA_PAD_MODEL") or P._cache_path()
        if not os.path.exists(path):
            self.skipTest("PAD model not cached")
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        p = P.score_crops([CROP, [30] * (80 * 80 * 3)], session=sess)
        self.assertEqual(len(p), 2)
        for v in p:
            self.assertTrue(0.0 <= v <= 1.0)


if __name__ == "__main__":
    unittest.main()
