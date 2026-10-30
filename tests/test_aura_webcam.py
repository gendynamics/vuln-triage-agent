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


def _att_frames(n=40, look=0.1, blink=0.05, yaw=0.0, seed=0):
    rng = random.Random(seed); out = []
    for i in range(n):
        j = lambda: max(0.0, min(1.0, rng.gauss(0, 0.03)))
        b = {k: j() for k in ("eyeLookInLeft", "eyeLookUpLeft", "eyeLookDownLeft", "eyeLookInRight", "eyeLookUpRight", "eyeLookDownRight")}
        b["eyeLookOutLeft"] = look + j(); b["eyeLookOutRight"] = j()
        b["eyeLookInRight"] = look + j()  # both eyes looking to the same side
        b["eyeBlinkLeft"] = b["eyeBlinkRight"] = blink
        out.append({"t_ms": i * 33, "blend": b, "yaw": yaw, "pitch": 0.0})
    return out


class AttentionTests(unittest.TestCase):
    def test_attentive(self):
        self.assertEqual(W.attention_channel(_att_frames()), (0.5, []))

    def test_looking_away(self):
        llr, rs = W.attention_channel(_att_frames(look=0.8))
        self.assertLess(llr, 0); self.assertIn("Inattentive", rs)

    def test_eyes_closed(self):
        llr, rs = W.attention_channel(_att_frames(blink=0.9))
        self.assertIn("EyesClosed", rs); self.assertIn("Inattentive", rs)

    def test_head_turned_and_no_data(self):
        self.assertIn("Inattentive", W.attention_channel(_att_frames(yaw=50))[1])
        self.assertEqual(W.attention_channel([{"t_ms": 0, "blend": {"jawOpen": 0.1}}] * 30), (0.0, []))


def _cons_frames(n=60, noise=0.0, teleport_at=(), dt=33, gap_at=None, seed=0):
    rng = random.Random(seed); out = []; t = 0
    for i in range(n):
        t += 400 if i == gap_at else dt
        x = 300 + 5 * math.sin(i / 8) + (200 if any(i >= k and i < k + 3 for k in teleport_at) else 0)
        jaw = 0.3 + 0.25 * math.sin(i / 5) + rng.gauss(0, noise)
        out.append({"t_ms": t, "iris_l": (x - 60, 240.0), "iris_r": (x + 60, 240.0),
                    "blend": {"jawOpen": jaw, "mouthSmileLeft": 0.1, "mouthSmileRight": 0.1}})
    return out


class ConsistencyTests(unittest.TestCase):
    def test_smooth_live(self):
        self.assertEqual(W.consistency_channel(_cons_frames()), (0.5, []))

    def test_teleport(self):
        llr, rs = W.consistency_channel(_cons_frames(teleport_at=(20,)))  # jump there and back = 2
        self.assertIn("FrameTeleport", rs); self.assertLess(llr, 0)

    def test_jitter(self):
        self.assertIn("BlendshapeJitter", W.consistency_channel(_cons_frames(noise=0.2))[1])

    def test_gap(self):
        self.assertIn("IrregularTiming", W.consistency_channel(_cons_frames(gap_at=30))[1])


# 3D face points (x, y, z) in IOD units: nose tip/bridge forward, cheeks back
_FACE3D = [(0, 0.45, -0.45), (0, -0.1, -0.2), (-0.65, 0, 0), (0.65, 0, 0), (-0.2, 0, -0.05), (0.2, 0, -0.05),
           (-0.35, 0.9, -0.1), (0.35, 0.9, -0.1), (0, 1.4, -0.05), (0, -0.7, -0.05), (-0.95, 0.4, 0.35), (0.95, 0.4, 0.35)]


def _proj_frames(flat=False, yaw_max=20.0, n=30, noise=0.002, seed=0):
    rng = random.Random(seed); out = []
    for i in range(n):
        yaw = yaw_max * math.sin(i / n * math.pi)
        c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        pts = []
        for x, y, z in _FACE3D:
            z = 0.0 if flat else z
            if flat:  # a photo turned like a card: rotate the plane itself
                X, Z = c * x + s_ * z, -s_ * x + c * z
            else:
                X, Z = c * x + s_ * z, -s_ * x + c * z
            f = 600 / (6 + Z)
            pts.append((320 + 100 * X * f / 100 + rng.gauss(0, noise * 100), 240 + 100 * y * f / 100 + rng.gauss(0, noise * 100)))
        il, ir = pts[2], pts[3]
        out.append({"t_ms": i * 33, "mesh2d": pts, "yaw": yaw, "iris_l": il, "iris_r": ir})
    return out


class PlanarityTests(unittest.TestCase):
    def test_homography_exact(self):
        src = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.3)]
        dst = [((2 * x + 1) / (0.1 * x + 1), (3 * y + x) / (0.1 * x + 1)) for x, y in src]
        H = W.fit_homography(src, dst)
        for p, q in zip(src, dst):
            self.assertLess(math.dist(H(p), q), 1e-6)

    def test_real_face_has_parallax(self):
        llr, rs, vs = W.planarity_channel(_proj_frames())
        self.assertEqual(vs, []); self.assertGreater(llr, 0)

    def test_rotated_photo_is_planar(self):
        llr, rs, vs = W.planarity_channel(_proj_frames(flat=True))
        self.assertIn("PlanarFace", vs)

    def test_static_pose_inconclusive(self):
        self.assertEqual(W.planarity_channel(_proj_frames(flat=True, yaw_max=3))[1], ["PoseTooStatic3D"])


def _rppg_frames(pulse=True, base=140.0, noise=0.2, hz=1.2, n=270, fps=30, seed=0):
    rng = random.Random(seed); out = []
    for i in range(n):
        t = i / fps; p = math.sin(2 * math.pi * hz * t) if pulse else 0.0
        roi = lambda: [base * (1 + 0.0006 * p) + rng.gauss(0, noise), base * 0.8 * (1 + 0.002 * p) + rng.gauss(0, noise),
                       base * 0.6 * (1 + 0.0012 * p) + rng.gauss(0, noise)]
        out.append({"t_ms": int(t * 1000), "rppg": {k: roi() for k in W.ROIS}})
    return out


class RppgTests(unittest.TestCase):
    def test_pulse_detected_and_rate(self):
        f = W.rppg_features(_rppg_frames(seed=1))
        self.assertAlmostEqual(f["hz"], 1.2, delta=0.1); self.assertEqual(f["coherent"], 3)
        self.assertEqual(W.rppg_channel(_rppg_frames(seed=1)), (1.5, [], []))

    def test_no_pulse_veto_in_good_light(self):
        vetoed = sum("NoPulse" in W.rppg_channel(_rppg_frames(False, seed=s))[2] for s in range(10))
        self.assertGreaterEqual(vetoed, 6)

    def test_pulse_never_vetoed(self):
        for s in range(10):
            self.assertNotIn("NoPulse", W.rppg_channel(_rppg_frames(seed=s))[2])

    def test_low_light_inconclusive(self):
        self.assertEqual(W.rppg_channel(_rppg_frames(False, base=30.0)), (0.0, ["LowLight"], []))

    def test_short_or_slow(self):
        self.assertEqual(W.rppg_channel(_rppg_frames(n=60))[1], ["RppgShort"])
        self.assertEqual(W.rppg_channel(_rppg_frames(n=60, fps=8))[1], ["RppgShort"])

    def test_pos_chrom_recover_tone(self):
        rgb = [(100 * (1 + 0.003 * math.sin(2 * math.pi * 1.5 * i / 30)), 80 * (1 + 0.01 * math.sin(2 * math.pi * 1.5 * i / 30)), 60.0) for i in range(240)]
        for sig in (W.pos_signal(rgb), W.chrom_signal(rgb)):
            hz, snr = W.spectrum_snr(W._detrend(sig))
            self.assertAlmostEqual(hz, 1.5, delta=0.05); self.assertGreater(snr, 5)


class ConsistencyFpsTests(unittest.TestCase):
    def test_low_fps_skips_roughness(self):
        self.assertNotIn("BlendshapeJitter", W.consistency_channel(_cons_frames(noise=0.2, dt=120))[1])
