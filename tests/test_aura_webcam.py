"""Synthetic tests for the webcam-only channels in src/aura_webcam.py."""
import math, os, random, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import aura_webcam as W


def lips(open_amt, w=60.0, cx=320.0, cy=400.0, roll=0.0):
    """20-point outer/inner lip rings (aura_scan index order), mouth opening `open_amt` px."""
    def ring(rw, rh):
        pts = []
        for k in range(20):
            a = math.pi - 2 * math.pi * k / 20 * -1  # start at left corner (61), go along lower lip
            x, y = rw * math.cos(a), rh * math.sin(a)
            pts.append((cx + x * math.cos(roll) - y * math.sin(roll), cy + x * math.sin(roll) + y * math.cos(roll)))
        return pts
    return ring(w / 2, 6 + open_amt), ring(w / 2.6, 0.5 + open_amt * 0.8)


def extras(kind="talk", n=60, seed=0):
    rng = random.Random(seed)
    out = []
    for k in range(n):
        o = 8 * (0.5 + 0.5 * math.sin(k / 3.0)) if kind == "talk" else 0.0
        if kind == "erratic":
            o = rng.choice([0, 15])
        lo, li = lips(o + rng.gauss(0, 0.05))
        out.append({"t_ms": 33 * k, "lips_outer": lo, "lips_inner": li, "iris_l": (290, 300), "iris_r": (350, 300),
                    "yaw": 0.0, "blend": {"jawOpen": o / 8.0 if kind == "talk" else 0.0}, "lip_z": [0] * 6, "iod_px": 60.0})
    return out


class TestGeometry(unittest.TestCase):
    def test_procrustes_removes_roll_and_scale(self):
        a = W.procrustes([(10, 0)], (0, 0), (20, 0))
        b = W.procrustes([(0, 20)], (0, 0), (0, 40))  # rotated 90°, scaled ×2
        self.assertAlmostEqual(a[0][0], b[0][0], places=6); self.assertAlmostEqual(a[0][1], b[0][1], places=6)

    def test_mar_increases_with_opening(self):
        c = W.lip_geometry(*lips(0)); o = W.lip_geometry(*lips(10))
        self.assertGreater(o["mar"], c["mar"]); self.assertGreater(o["inner_hw"], c["inner_hw"])

    def test_talking_positive_static_negative(self):
        llr_t, _ = W.geometry_channel(extras("talk"))
        llr_s, rs = W.geometry_channel(extras("static"))
        self.assertGreater(llr_t, 0); self.assertLess(llr_s, 0); self.assertIn("GeometryStatic", rs)

    def test_deltas(self):
        d1, d2 = W.deltas([0, 1, 2, 3, 4])
        self.assertEqual(len(d1), 4); self.assertEqual(len(d2), 3)


if __name__ == "__main__":
    unittest.main()


# ---------------- calibration tool ----------------
import importlib.util as _ilu, os as _os, json as _json
_spec = _ilu.spec_from_file_location("calibrate", _os.path.join(_os.path.dirname(__file__), "..", "tools", "calibrate.py"))
CT = _ilu.module_from_spec(_spec); _spec.loader.exec_module(CT)


class CalibrationToolTests(unittest.TestCase):
    def test_platt_recovers_gaussian_optimum(self):
        res = CT.evaluate(CT.synthetic(2000, seed=3))
        # live N(14,5), spoof N(-4,5): optimal a = 18/25 = 0.72, b = -(196-16)/50 = -3.6
        assert abs(res["platt_a"] - 0.72) < 0.1 and abs(res["platt_b"] + 3.6) < 0.6
        assert res["cv"]["roc_auc"] > 0.98


    def test_clopper_pearson_rule_of_three(self):
        assert abs(CT.clopper_pearson_upper(0, 300000) - 1e-5) < 1e-6
        assert abs(CT.clopper_pearson_upper(0, 3000) - 1e-3) < 1e-4
        assert CT.clopper_pearson_upper(5, 100) > 0.05


    def test_cfg_roundtrip_and_server_recording(self):
        import tempfile, pathlib; tmp_path = pathlib.Path(tempfile.mkdtemp())
        import aura_scan as S
        res = CT.evaluate(CT.synthetic(200), isotonic=True)
        cfg = tmp_path / "c.cfg"; CT.write_cfg(res, str(cfg))
        cal = S.Calibration.from_config(cfg.read_text())
        assert cal.calibrated and abs(cal.platt_a - res["platt_a"]) < 1e-5 and len(cal.isotonic) >= 2
        assert cal.posterior(30) > 0.99 and cal.posterior(-20) < 0.01
        jl = tmp_path / "c.jsonl"
        jl.write_text("\n".join(_json.dumps(r) for r in CT.synthetic(40)) + "\n")
        assert CT.main(["--in", str(jl), "--out", str(tmp_path / "o.cfg")]) == 0 and (tmp_path / "o.cfg").exists()
