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
