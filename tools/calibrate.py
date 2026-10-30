#!/usr/bin/env python3
"""Fit AURA's posterior calibration from consented calibration recordings.

Input : app/data/calibration.jsonl  (one line per session: label + evidence LLRs, no images/landmarks)
Output: app/data/calibration.cfg    (platt_a / platt_b [/ isotonic] — read by app/server.py at start)

  python tools/calibrate.py [--in FILE] [--out FILE] [--isotonic] [--folds 5]
  python tools/calibrate.py --synthetic 400     # self-test on synthetic data

Method: Platt scaling = 1-D logistic regression of label (live=1, any spoof=0) on the summed
evidence LLR (total_llr), fitted by Newton's method; optional isotonic (PAV) on top.
Metrics are k-fold cross-validated (each session scored by a model that never saw it):
FAR (spoof accepted) / FRR (live rejected) at the 0.85 legacy and 0.999 high-assurance tiers,
ROC AUC, EER, and a Clopper-Pearson 95% upper bound on FAR.

Sample sizes (rule of three, zero errors, 95%): showing FAR <= 1e-5 needs ~300,000 spoof attempts
with no accepts; FAR <= 1e-6 ~3,000,000; FRR <= 0.1% (99.9% live acceptance) ~3,000 live attempts.
"""
from __future__ import annotations

import argparse, json, math, os, random, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = {"live", "printed_photo", "screen_replay", "deepfake", "mask"}
TIERS = (0.85, 0.999)


def sig(z):
    return 1 / (1 + math.exp(-z)) if z > -700 else 0.0


def fit_platt(x, y, iters=100, l2=1e-3):
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for xi, yi in zip(x, y):
            p = sig(a * xi + b); w = p * (1 - p); e = p - yi
            ga += e * xi; gb += e; haa += w * xi * xi; hab += w * xi; hbb += w
        ga += l2 * a; haa += l2; hbb += 1e-9
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        step = max(abs(da), abs(db), 1e-12); k = min(1.0, 2.0 / step)  # damped Newton
        a, b = a - k * da, b - k * db
        if abs(da) + abs(db) < 1e-10:
            break
    return a, b


def fit_isotonic(p, y):
    """Pool-adjacent-violators → list of (p, calibrated) knots."""
    pts = sorted(zip(p, y))
    blocks = [[v, float(t), 1] for v, t in pts]  # [x, sum_y, n]
    out = []
    for blk in blocks:
        out.append(blk)
        while len(out) > 1 and out[-2][1] / out[-2][2] > out[-1][1] / out[-1][2]:
            x2, s2, n2 = out.pop(); out[-1][1] += s2; out[-1][2] += n2; out[-1][0] = x2
    return [(round(x, 6), round(s / n, 6)) for x, s, n in out]


def apply_iso(knots, p):
    if len(knots) < 2:
        return p
    if p <= knots[0][0]:
        return knots[0][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if p <= x1:
            return y0 + (p - x0) / max(x1 - x0, 1e-12) * (y1 - y0)
    return knots[-1][1]


def binom_cdf(k, n, p):
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    lp, lq, s = math.log(p), math.log1p(-p), 0.0
    for i in range(k + 1):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(s, 1.0)


def clopper_pearson_upper(k, n, conf=0.95):
    """One-sided upper bound: largest p with P(X <= k | n, p) >= 1-conf."""
    if n == 0:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


def roc(scores, y):
    pos = sum(y); neg = len(y) - pos
    pts, tp, fp = [(0.0, 0.0)], 0, 0
    for s, t in sorted(zip(scores, y), key=lambda z: -z[0]):
        tp += t; fp += 1 - t
        pts.append((fp / max(neg, 1), tp / max(pos, 1)))
    auc = sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    eer = min(pts, key=lambda q: abs(q[0] - (1 - q[1])))
    return auc, (eer[0] + 1 - eer[1]) / 2


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("label") in LABELS and isinstance(r.get("evidence"), dict):
                rows.append(r)
    return rows


def synthetic(n, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        live = i % 2 == 0
        llr = rng.gauss(14, 5) if live else rng.gauss(-4, 5)
        rows.append({"label": "live" if live else rng.choice(sorted(LABELS - {"live"})),
                     "evidence": {"total_llr": llr}, "vetoes": [] if live or rng.random() < 0.5 else ["StaticMesh"]})
    return rows


def evaluate(rows, folds=5, isotonic=False, seed=0):
    x = [float(r["evidence"].get("total_llr", 0.0)) for r in rows]
    y = [1 if r["label"] == "live" else 0 for r in rows]
    veto = [bool(r.get("vetoes")) for r in rows]
    idx = list(range(len(rows))); random.Random(seed).shuffle(idx)
    cv = [0.0] * len(rows)
    for k in range(folds):
        test = set(idx[k::folds]); tr = [i for i in idx if i not in test]
        a, b = fit_platt([x[i] for i in tr], [y[i] for i in tr])
        knots = fit_isotonic([sig(a * x[i] + b) for i in tr], [y[i] for i in tr]) if isotonic else []
        for i in test:
            cv[i] = apply_iso(knots, sig(a * x[i] + b))
    a, b = fit_platt(x, y)
    knots = fit_isotonic([sig(a * xi + b) for xi in x], y) if isotonic else []
    n_spoof, n_live = y.count(0), y.count(1)
    tiers = {}
    for t in TIERS:
        fa = sum(1 for p, yi, v in zip(cv, y, veto) if yi == 0 and p >= t and not v)
        fr = sum(1 for p, yi, v in zip(cv, y, veto) if yi == 1 and (p < t or v))
        tiers[str(t)] = {"FAR": fa / max(n_spoof, 1), "false_accepts": fa, "spoof_n": n_spoof,
                         "FAR_upper95_clopper_pearson": clopper_pearson_upper(fa, n_spoof),
                         "FRR": fr / max(n_live, 1), "false_rejects": fr, "live_n": n_live}
    auc, eer = roc(cv, y)
    return {"platt_a": a, "platt_b": b, "isotonic": knots, "n": len(rows), "folds": folds,
            "cv": {"roc_auc": auc, "eer": eer, "tiers": tiers},
            "needed_for_claims": {"FAR<=1e-5 (95%, 0 accepts)": math.ceil(3 / 1e-5),
                                  "FAR<=1e-6 (95%, 0 accepts)": math.ceil(3 / 1e-6),
                                  "FRR<=1e-3 (95%, 0 rejects)": math.ceil(3 / 1e-3)}}


def write_cfg(res, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"# fitted by tools/calibrate.py on n={res['n']} sessions ({res['folds']}-fold CV AUC {res['cv']['roc_auc']:.4f})\n")
        fh.write(f"platt_a = {res['platt_a']:.6f}\nplatt_b = {res['platt_b']:.6f}\n")
        if res["isotonic"]:
            fh.write("isotonic = " + ",".join(f"{p}:{q}" for p, q in res["isotonic"]) + "\n")
        fh.write("calibrated = true\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=os.path.join(ROOT, "app", "data", "calibration.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "app", "data", "calibration.cfg"))
    ap.add_argument("--isotonic", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--synthetic", type=int, default=0, help="self-test on N synthetic sessions (writes nothing unless --out given explicitly)")
    a = ap.parse_args(argv)
    rows = synthetic(a.synthetic) if a.synthetic else load(a.inp)
    labels = {l: sum(1 for r in rows if r["label"] == l) for l in sorted(LABELS)}
    if labels.get("live", 0) < 2 or len(rows) - labels.get("live", 0) < 2:
        print(json.dumps({"error": "need >= 2 live and >= 2 spoof sessions", "labels": labels})); return 2
    res = evaluate(rows, a.folds, a.isotonic)
    res["labels"] = labels
    if not a.synthetic or "--out" in (argv or sys.argv):
        write_cfg(res, a.out); res["wrote"] = a.out
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
