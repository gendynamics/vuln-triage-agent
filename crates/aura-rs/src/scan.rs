//! Silent 5-second liveness scan with evidence-accumulation scoring.
//!
//! Consumes per-frame face *landmarks* plus grayscale eye-region pixels from
//! an upstream detector (e.g. MediaPipe Face Mesh with iris refinement). This
//! crate bundles no detector and renders nothing: the "orange shapes locking
//! onto the eyes" and the "mesh tracing the lips" exist only as internal data
//! (`EyeLock`, `LipMesh`).
//!
//! Liveness, not identity: nothing here identifies or matches a person. The
//! result carries only scores, a status and reason codes; frame buffers are
//! zeroed and dropped once scoring finishes.
//!
//! Glint-arc physics: the corneal glint (first Purkinje image) of a fixed
//! light moves roughly half as far as the iris when a real eye rotates, since
//! the corneal radius (~7.8 mm) is smaller than the eye's rotation radius
//! (~13 mm). A glint printed on a photo or shown on a flat screen moves 1:1
//! with the iris. The arc ratio is the least-squares slope of glint
//! displacement on iris displacement (eye-relative, in eye widths): ~0.5 live,
//! ~1.0 flat.
//!
//! IMPORTANT: every threshold and likelihood parameter below is a hand-set,
//! UNCALIBRATED design value. The 99.9 % high-assurance tier and the 1e-5
//! false-accept figure are design targets, not measured accuracy; they must
//! be validated on a real labelled anti-spoofing dataset.

use crate::{fuse, GlintReading, LipsyncReading, LIPSYNC_ACCEPT, TARGET_FALSE_ACCEPT};
use serde::Serialize;

pub const WINDOW_MS: u64 = 5_000;
pub const MIN_FRAMES: usize = 30;
pub const MIN_SPAN_MS: u64 = 3_000;
pub const CLOSE_EYE_WIDTH_PX: f64 = 60.0;
pub const MIN_IRIS_MOTION: f64 = 0.01;
pub const MIN_LOCK_FRACTION: f64 = 0.6;
pub const STATIC_MESH_EPS: f64 = 0.004;
pub const MAX_DUP_FRACTION: f64 = 0.10;
pub const GLINT_MIN_LEVEL: u8 = 180;
/// Glint-ratio at or above which the cornea is judged flat (photo/screen).
pub const FLAT_GLINT_RATIO: f64 = 0.85;
/// Affine-fit residual (eye widths) below which all landmarks move as one plane.
pub const PLANAR_RESIDUAL_EPS: f64 = 0.003;
/// Natural range of eye-landmark 2nd-difference jitter (eye widths / frame²).
pub const JITTER_MIN: f64 = 0.0015;
pub const JITTER_MAX: f64 = 0.06;
pub const MOIRE_ROW_SCORE: f64 = 0.5;
pub const FLICKER_ALT_FRACTION: f64 = 0.6;
pub const FLICKER_MIN_STEP: f64 = 1.5;
pub const AV_MAX_LAG_MS: f64 = 120.0;
pub const AV_MIN_CORR: f64 = 0.3;
pub const LIGHT_MIN_COS: f64 = 0.5;
/// Legacy (weighted-average) decision threshold.
pub const LEGACY_THRESHOLD: f64 = 0.85;
/// High-assurance posterior threshold.
pub const HIGH_ASSURANCE_THRESHOLD: f64 = 0.999;

/// Minimum accumulated log-likelihood ratio for high-assurance accept,
/// derived from TARGET_FALSE_ACCEPT via Wald's bound: accepting only when
/// LR >= 1/alpha keeps P(accept | spoof) <= alpha, *if* the LLRs are calibrated
/// and the channels conditionally independent.
pub fn llr_accept_threshold() -> f64 {
    (1.0 / TARGET_FALSE_ACCEPT).ln()
}

pub type Pt = (f64, f64);

#[derive(Clone, Debug)]
pub struct EyePatch {
    pub origin: Pt,
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct EyeObs {
    pub contour: Vec<Pt>,
    pub iris_center: Pt,
    pub iris_radius: f64,
    pub patch: EyePatch,
}

#[derive(Clone, Debug)]
pub struct FrameObs {
    pub t_ms: u64,
    pub left_eye: EyeObs,
    pub right_eye: EyeObs,
    /// Outer and inner lip contours, same point count, same winding.
    pub lips_outer: Vec<Pt>,
    pub lips_inner: Vec<Pt>,
    /// Optional expected mouth openness from an audio viseme track, aligned to
    /// this frame (upstream speech model). Enables the A/V sync checks.
    pub audio_openness: Option<f64>,
}

/// Internal lock state for one eye on one frame (the video's orange shape).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EyeLock {
    Searching,
    Locked { glint: Pt },
}

/// Glint detection detail used by the shape checks.
#[derive(Clone, Copy, Debug)]
pub struct GlintObs {
    pub lock: EyeLock,
    pub components: usize,
    pub area: usize,
    pub bbox_fill: f64,
}

/// Triangulated lip ring between outer and inner contour (the video's mesh).
#[derive(Clone, Debug)]
pub struct LipMesh {
    pub vertices: Vec<Pt>,
    pub triangles: Vec<[usize; 3]>,
    pub width_px: f64,
    pub openness: f64,
    /// Inner/outer width ratio: lip-corner spread (rounded vs spread visemes).
    pub spread: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ScanStatus { Pass, Fail, Inconclusive }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ReasonCode {
    NoConsent,
    TooFewFrames,
    WindowTooShort,
    TimestampNonMonotonic,
    // --- vetoes: replay / static ---
    DuplicateFrames,
    StaticMesh,
    // --- vetoes: printed photo ---
    FlatGlintGeometry,
    PlanarRigidMotion,
    // --- vetoes: screen replay ---
    MoirePattern,
    ScreenFlicker,
    RectangularGlint,
    DoubledGlint,
    // --- vetoes: deepfake / synthesis ---
    AvDesync,
    JitterOutOfRange,
    GlintLightMismatch,
    /// Lip-reading challenge-response failed (see `challenge`).
    ChallengeFailed,
    // --- soft findings (affect score, not vetoes) ---
    NoOcularMotion,
    GlintNotLocked,
    BinocularInconsistent,
    NoBlinkObserved,
    WeakArticulation,
    BelowHighAssurance,
    FusedBelowThreshold,
    Passed,
}

impl ReasonCode {
    pub fn is_veto(self) -> bool {
        use ReasonCode::*;
        matches!(self, TimestampNonMonotonic | DuplicateFrames | StaticMesh | FlatGlintGeometry
            | PlanarRigidMotion | MoirePattern | ScreenFlicker | RectangularGlint | DoubledGlint
            | AvDesync | JitterOutOfRange | GlintLightMismatch | ChallengeFailed)
    }
}

/// Which decision sets `status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Tier { Legacy, HighAssurance }

/// Platt / isotonic calibration of the accumulated LLR to a probability.
/// Defaults are an identity Platt map and are explicitly UNCALIBRATED.
#[derive(Clone, Debug, Serialize)]
pub struct Calibration {
    pub platt_a: f64,
    pub platt_b: f64,
    /// Optional isotonic table of (raw posterior, calibrated posterior), sorted.
    pub isotonic: Vec<(f64, f64)>,
    pub calibrated: bool,
    /// Prior log-odds of a live subject.
    pub prior_log_odds: f64,
}

impl Default for Calibration {
    fn default() -> Self {
        Calibration { platt_a: 1.0, platt_b: 0.0, isotonic: Vec::new(), calibrated: false, prior_log_odds: 0.0 }
    }
}

impl Calibration {
    /// Parse `key=value` lines: platt_a, platt_b, prior_log_odds, calibrated
    /// (true/false), isotonic=x:y,x:y,... Unknown keys are ignored.
    pub fn from_config(text: &str) -> Result<Self, String> {
        let mut c = Calibration::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let (k, v) = line.split_once('=').ok_or_else(|| format!("bad line: {line}"))?;
            let (k, v) = (k.trim(), v.trim());
            let num = |v: &str| v.parse::<f64>().map_err(|e| format!("{k}: {e}"));
            match k {
                "platt_a" => c.platt_a = num(v)?,
                "platt_b" => c.platt_b = num(v)?,
                "prior_log_odds" => c.prior_log_odds = num(v)?,
                "calibrated" => c.calibrated = v == "true",
                "isotonic" => {
                    c.isotonic = v.split(',').map(|p| {
                        let (x, y) = p.split_once(':').ok_or("isotonic pair")?;
                        Ok((x.trim().parse::<f64>().map_err(|e| e.to_string())?, y.trim().parse::<f64>().map_err(|e| e.to_string())?))
                    }).collect::<Result<_, String>>()?;
                }
                _ => {}
            }
        }
        Ok(c)
    }

    pub fn posterior(&self, llr_total: f64) -> f64 {
        let z = self.platt_a * (llr_total + self.prior_log_odds) + self.platt_b;
        let p = 1.0 / (1.0 + (-z).exp());
        if self.isotonic.len() < 2 { return p; }
        let t = &self.isotonic;
        if p <= t[0].0 { return t[0].1; }
        for w in t.windows(2) {
            if p <= w[1].0 {
                let f = (p - w[0].0) / (w[1].0 - w[0].0).max(1e-12);
                return w[0].1 + f * (w[1].1 - w[0].1);
            }
        }
        t[t.len() - 1].1
    }
}

/// Per-channel evidence (natural-log likelihood ratios, live vs spoof).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Evidence {
    pub glint_llr: f64,
    pub binocular_llr: f64,
    pub glint_stability_llr: f64,
    pub lip_llr: f64,
    pub av_llr: f64,
    pub blink_llr: f64,
    pub jitter_llr: f64,
    pub w_glint: f64,
    pub w_lip: f64,
    pub total_llr: f64,
}

/// Scan result: scores and codes only. No pixels, no landmarks, no template.
#[derive(Clone, Debug, Serialize)]
pub struct ScanResult {
    pub status: ScanStatus,
    pub tier: Tier,
    /// Legacy weighted-average fused score (unchanged `fuse`).
    pub fused_score: f64,
    pub legacy_pass: bool,
    /// Calibrated posterior from evidence accumulation.
    pub posterior: f64,
    pub high_assurance_pass: bool,
    pub calibrated: bool,
    pub evidence: Evidence,
    pub glint_score: f64,
    pub glint_arc_ratio: f64,
    pub lipsync_score: f64,
    pub viseme_match: f64,
    pub distance_hint: String,
    pub frames_used: usize,
    pub glint_lock_fraction: [f64; 2],
    pub vetoes: Vec<ReasonCode>,
    pub reasons: Vec<ReasonCode>,
}

impl ScanResult {
    fn early(status: ScanStatus, n: usize, r: ReasonCode) -> Self {
        ScanResult {
            status, tier: Tier::HighAssurance, fused_score: 0.0, legacy_pass: false, posterior: 0.0,
            high_assurance_pass: false, calibrated: false, evidence: Evidence::default(),
            glint_score: 0.0, glint_arc_ratio: 0.0, lipsync_score: 0.0, viseme_match: 0.0,
            distance_hint: "unknown".into(), frames_used: n, glint_lock_fraction: [0.0, 0.0],
            vetoes: if r.is_veto() { vec![r] } else { vec![] }, reasons: vec![r],
        }
    }
}

// ---------- geometry helpers ----------

fn centroid(p: &[Pt]) -> Pt {
    let n = p.len().max(1) as f64;
    let (sx, sy) = p.iter().fold((0.0, 0.0), |a, q| (a.0 + q.0, a.1 + q.1));
    (sx / n, sy / n)
}

fn extent(p: &[Pt], y: bool) -> f64 {
    if p.is_empty() { return 0.0; }
    let v = |q: &Pt| if y { q.1 } else { q.0 };
    let lo = p.iter().map(v).fold(f64::INFINITY, f64::min);
    let hi = p.iter().map(v).fold(f64::NEG_INFINITY, f64::max);
    hi - lo
}

fn polygon_area(p: &[Pt]) -> f64 {
    let n = p.len();
    if n < 3 { return 0.0; }
    let mut a = 0.0;
    for i in 0..n {
        let (x0, y0) = p[i];
        let (x1, y1) = p[(i + 1) % n];
        a += x0 * y1 - x1 * y0;
    }
    (a * 0.5).abs()
}

fn mean(v: &[f64]) -> f64 { if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 } }

fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn pearson_lag(a: &[f64], b: &[f64], lag: i64) -> f64 {
    let n = a.len() as i64;
    let (mut xa, mut xb) = (Vec::new(), Vec::new());
    for i in 0..n {
        let j = i + lag;
        if j >= 0 && j < n { xa.push(a[i as usize]); xb.push(b[j as usize]); }
    }
    let (ma, mb) = (mean(&xa), mean(&xb));
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (x, y) in xa.iter().zip(xb.iter()) {
        num += (x - ma) * (y - mb); da += (x - ma).powi(2); db += (y - mb).powi(2);
    }
    if da <= 0.0 || db <= 0.0 { 0.0 } else { num / (da * db).sqrt() }
}

/// Brightest blob(s) inside the iris disc → intensity-weighted centroid, plus
/// component count and bounding-box fill for the glint-shape spoof checks.
pub fn detect_glint(eye: &EyeObs) -> GlintObs {
    let none = GlintObs { lock: EyeLock::Searching, components: 0, area: 0, bbox_fill: 0.0 };
    let p = &eye.patch;
    if p.pixels.len() != p.width * p.height || eye.iris_radius <= 0.0 { return none; }
    let (cx, cy) = (eye.iris_center.0 - p.origin.0, eye.iris_center.1 - p.origin.1);
    let r2 = eye.iris_radius * eye.iris_radius;
    let inside = |x: usize, y: usize| { let (dx, dy) = (x as f64 - cx, y as f64 - cy); dx * dx + dy * dy <= r2 };
    let mut peak = 0u8;
    let mut n_in = 0usize;
    for y in 0..p.height { for x in 0..p.width {
        if inside(x, y) { n_in += 1; peak = peak.max(p.pixels[y * p.width + x]); }
    }}
    if n_in == 0 || peak < GLINT_MIN_LEVEL { return none; }
    let thr = (peak as f64 * 0.85).max(GLINT_MIN_LEVEL as f64);
    let mut mask = vec![false; p.width * p.height];
    let (mut sw, mut sx, mut sy, mut cnt) = (0.0, 0.0, 0.0, 0usize);
    let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0, usize::MAX, 0);
    for y in 0..p.height { for x in 0..p.width {
        let v = p.pixels[y * p.width + x] as f64;
        if inside(x, y) && v >= thr {
            mask[y * p.width + x] = true;
            sw += v; sx += v * x as f64; sy += v * y as f64; cnt += 1;
            x0 = x0.min(x); x1 = x1.max(x); y0 = y0.min(y); y1 = y1.max(y);
        }
    }}
    if cnt == 0 || cnt as f64 > 0.15 * n_in as f64 { return none; }
    // 4-connected components of size >= 2.
    let mut seen = vec![false; mask.len()];
    let mut comps = 0usize;
    for start in 0..mask.len() {
        if !mask[start] || seen[start] { continue; }
        let mut stack = vec![start];
        seen[start] = true;
        let mut size = 0;
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y) = (i % p.width, i / p.width);
            let mut nb = Vec::with_capacity(4);
            if x > 0 { nb.push(i - 1); }
            if x + 1 < p.width { nb.push(i + 1); }
            if y > 0 { nb.push(i - p.width); }
            if y + 1 < p.height { nb.push(i + p.width); }
            for j in nb { if mask[j] && !seen[j] { seen[j] = true; stack.push(j); } }
        }
        if size >= 2 { comps += 1; }
    }
    let bbox = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
    GlintObs {
        lock: EyeLock::Locked { glint: (sx / sw + p.origin.0, sy / sw + p.origin.1) },
        components: comps, area: cnt, bbox_fill: cnt as f64 / bbox,
    }
}

/// Backward-compatible wrapper returning only the lock state.
pub fn find_glint(eye: &EyeObs) -> EyeLock { detect_glint(eye).lock }

/// Triangulate the lip ring and measure width, openness and spread.
pub fn build_lip_mesh(outer: &[Pt], inner: &[Pt]) -> LipMesh {
    let n = outer.len().min(inner.len());
    let mut vertices = Vec::with_capacity(2 * n);
    vertices.extend_from_slice(&outer[..n]);
    vertices.extend_from_slice(&inner[..n]);
    let mut triangles = Vec::with_capacity(2 * n);
    for i in 0..n {
        let j = (i + 1) % n;
        triangles.push([i, j, n + i]);
        triangles.push([j, n + j, n + i]);
    }
    let width_px = extent(outer, false);
    let openness = if width_px > 0.0 { polygon_area(inner) / (width_px * width_px) } else { 0.0 };
    let spread = if width_px > 0.0 { extent(inner, false) / width_px } else { 0.0 };
    LipMesh { vertices, triangles, width_px, openness, spread }
}

fn frame_signature(f: &FrameObs) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut eat = |b: &[u8]| for x in b { h ^= *x as u64; h = h.wrapping_mul(0x100000001b3); };
    for e in [&f.left_eye, &f.right_eye] {
        for q in e.contour.iter().chain(std::iter::once(&e.iris_center)) {
            eat(&q.0.to_bits().to_le_bytes()); eat(&q.1.to_bits().to_le_bytes());
        }
        eat(&e.patch.pixels);
    }
    for q in f.lips_outer.iter().chain(f.lips_inner.iter()) {
        eat(&q.0.to_bits().to_le_bytes()); eat(&q.1.to_bits().to_le_bytes());
    }
    h
}

/// Mesh-based viseme/articulation score (0..1) from the openness and spread
/// time series: range of motion, speech-like rate, smoothness, and the
/// natural coupling between opening and corner spread.
fn articulation_score(open: &[f64], spread: &[f64], t_s: &[f64]) -> f64 {
    let n = open.len();
    if n < 3 { return 0.0; }
    let lo = open.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = open.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = hi - lo;
    let range_s = (range / 0.06).min(1.0);
    let m = mean(open);
    let mut crossings = 0usize;
    for i in 1..n { if (open[i - 1] - m).signum() != (open[i] - m).signum() { crossings += 1; } }
    let dur = (t_s[n - 1] - t_s[0]).max(1e-6);
    let cps = crossings as f64 / 2.0 / dur;
    let rate_s = if (0.5..=8.0).contains(&cps) { 1.0 } else { 0.3 };
    let mut jumps = 0usize;
    for i in 1..n { if (open[i] - open[i - 1]).abs() > 0.5 * range.max(1e-9) { jumps += 1; } }
    let smooth_s = 1.0 - (jumps as f64 / n as f64).min(1.0);
    // Spread should vary a little with opening (lips are not a rigid stamp).
    let spread_s = if std_dev(spread) > 1e-4 || range < 1e-6 { 1.0 } else { 0.6 };
    (0.5 * range_s + 0.2 * rate_s + 0.2 * smooth_s + 0.1 * spread_s).clamp(0.0, 1.0)
}

/// Least-squares 2-D affine fit P0 -> Pk; returns RMS residual (px).
fn affine_residual(p0: &[Pt], pk: &[Pt]) -> f64 {
    let n = p0.len().min(pk.len());
    if n < 4 { return f64::INFINITY; }
    let mut m = [[0.0f64; 3]; 3];
    let (mut bx, mut by) = ([0.0f64; 3], [0.0f64; 3]);
    for i in 0..n {
        let v = [p0[i].0, p0[i].1, 1.0];
        for r in 0..3 { for c in 0..3 { m[r][c] += v[r] * v[c]; } bx[r] += v[r] * pk[i].0; by[r] += v[r] * pk[i].1; }
    }
    let det3 = |a: &[[f64; 3]; 3]| a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let d = det3(&m);
    if d.abs() < 1e-12 { return f64::INFINITY; }
    let solve = |b: &[f64; 3]| {
        let mut s = [0.0; 3];
        for c in 0..3 { let mut a = m; for r in 0..3 { a[r][c] = b[r]; } s[c] = det3(&a) / d; }
        s
    };
    let (sx, sy) = (solve(&bx), solve(&by));
    let mut e = 0.0;
    for i in 0..n {
        let (x, y) = p0[i];
        let ex = sx[0] * x + sx[1] * y + sx[2] - pk[i].0;
        let ey = sy[0] * x + sy[1] * y + sy[2] - pk[i].1;
        e += ex * ex + ey * ey;
    }
    (e / n as f64).sqrt()
}

fn all_landmarks(f: &FrameObs) -> Vec<Pt> {
    let mut v = Vec::new();
    for e in [&f.left_eye, &f.right_eye] { v.extend_from_slice(&e.contour); v.push(e.iris_center); }
    v.extend_from_slice(&f.lips_outer);
    v.extend_from_slice(&f.lips_inner);
    v
}

/// Periodic high-frequency energy along rows of an eye crop (moiré from
/// re-photographing a pixel grid). Returns the mean periodicity score of
/// textured rows, or None when the crop has no textured rows.
fn moire_score(p: &EyePatch) -> Option<f64> {
    if p.pixels.len() != p.width * p.height || p.width < 20 { return None; }
    let mut scores = Vec::new();
    for y in 0..p.height {
        let row: Vec<f64> = p.pixels[y * p.width..(y + 1) * p.width].iter().map(|&v| v as f64).collect();
        if std_dev(&row) < 5.0 { continue; }
        let m = mean(&row);
        let r: Vec<f64> = row.iter().map(|v| v - m).collect();
        let r0: f64 = r.iter().map(|v| v * v).sum();
        let ac = |lag: usize| r.iter().zip(r.iter().skip(lag)).map(|(a, b)| a * b).sum::<f64>() / r0;
        let mut best: f64 = 0.0;
        let mut prev = ac(1);
        for lag in 2..=8 {
            let cur = ac(lag);
            // A rise after a dip = periodic structure; monotone decay = edges/texture.
            if cur > prev && cur > 0.0 { best = best.max(cur); }
            prev = cur;
        }
        scores.push(best);
    }
    if scores.is_empty() { None } else { Some(mean(&scores)) }
}

fn eye_aspect(e: &EyeObs) -> f64 {
    let w = extent(&e.contour, false);
    if w <= 0.0 { 0.0 } else { extent(&e.contour, true) / w }
}

/// Accumulates frames for one 5-second window. Construct only after the
/// integrating app has obtained explicit user consent.
pub struct SilentScanner {
    start_ms: Option<u64>,
    frames: Vec<FrameObs>,
    non_monotonic: bool,
    calibration: Calibration,
    tier: Tier,
}

impl SilentScanner {
    /// Returns `None` unless `consent` is true.
    pub fn new(consent: bool) -> Option<Self> {
        Self::with_config(consent, Calibration::default(), Tier::HighAssurance)
    }

    pub fn with_config(consent: bool, calibration: Calibration, tier: Tier) -> Option<Self> {
        if !consent { return None; }
        Some(SilentScanner { start_ms: None, frames: Vec::new(), non_monotonic: false, calibration, tier })
    }

    /// Push one frame. Returns false once the window has closed (frame ignored).
    pub fn push(&mut self, f: FrameObs) -> bool {
        let start = *self.start_ms.get_or_insert(f.t_ms);
        if f.t_ms < start || f.t_ms - start > WINDOW_MS { return false; }
        if let Some(last) = self.frames.last() { if f.t_ms <= last.t_ms { self.non_monotonic = true; } }
        self.frames.push(f);
        true
    }

    pub fn window_closed(&self, now_ms: u64) -> bool {
        self.start_ms.map_or(false, |s| now_ms.saturating_sub(s) > WINDOW_MS)
    }

    /// Score the window, then wipe all buffers.
    pub fn finish(mut self) -> ScanResult {
        let r = score_frames(&self.frames, self.non_monotonic, &self.calibration, self.tier);
        self.wipe();
        r
    }

    fn wipe(&mut self) {
        for f in self.frames.iter_mut() {
            for e in [&mut f.left_eye, &mut f.right_eye] {
                e.patch.pixels.iter_mut().for_each(|p| *p = 0);
                e.contour.iter_mut().for_each(|p| *p = (0.0, 0.0));
                e.iris_center = (0.0, 0.0);
            }
            f.lips_outer.iter_mut().for_each(|p| *p = (0.0, 0.0));
            f.lips_inner.iter_mut().for_each(|p| *p = (0.0, 0.0));
            f.audio_openness = None;
        }
        self.frames.clear();
        self.frames.shrink_to_fit();
    }
}

impl Drop for SilentScanner { fn drop(&mut self) { self.wipe(); } }

/// Run a whole window (default calibration, high-assurance tier).
pub fn scan_window(consent: bool, frames: Vec<FrameObs>) -> ScanResult {
    scan_window_with(consent, frames, Calibration::default(), Tier::HighAssurance)
}

pub fn scan_window_with(consent: bool, frames: Vec<FrameObs>, cal: Calibration, tier: Tier) -> ScanResult {
    let mut s = match SilentScanner::with_config(consent, cal, tier) {
        Some(s) => s,
        None => return ScanResult::early(ScanStatus::Inconclusive, 0, ReasonCode::NoConsent),
    };
    for f in frames { if !s.push(f) { break; } }
    s.finish()
}

/// Gaussian log-likelihood ratio of glint ratio r: live N(0.5, s) vs flat N(1.0, s).
pub fn glint_llr(r: f64, sigma: f64) -> f64 {
    let s2 = 2.0 * sigma * sigma;
    (((r - 1.0).powi(2) - (r - 0.5).powi(2)) / s2).clamp(-20.0, 20.0)
}

/// Measurement noise of the glint ratio (heuristic, uncalibrated): a floor
/// plus a term shrinking with eye size (centroid error ~0.5 px over fewer px).
pub fn glint_sigma(eye_width_px: f64) -> f64 { 0.05 + 3.0 / eye_width_px.max(1.0) }

fn score_frames(frames: &[FrameObs], non_monotonic: bool, cal: &Calibration, tier: Tier) -> ScanResult {
    let n = frames.len();
    if non_monotonic { return ScanResult::early(ScanStatus::Fail, n, ReasonCode::TimestampNonMonotonic); }
    if n < MIN_FRAMES { return ScanResult::early(ScanStatus::Inconclusive, n, ReasonCode::TooFewFrames); }
    if frames[n - 1].t_ms - frames[0].t_ms < MIN_SPAN_MS {
        return ScanResult::early(ScanStatus::Inconclusive, n, ReasonCode::WindowTooShort);
    }
    use ReasonCode::*;
    let mut reasons: Vec<ReasonCode> = Vec::new();
    let mut ev = Evidence::default();
    let t_s: Vec<f64> = frames.iter().map(|f| f.t_ms as f64 / 1000.0).collect();
    let dt_ms = median(&t_s.windows(2).map(|w| (w[1] - w[0]) * 1000.0).collect::<Vec<_>>()).max(1.0);

    // Replay: exact duplicate consecutive frames.
    let sigs: Vec<u64> = frames.iter().map(frame_signature).collect();
    let dups = sigs.windows(2).filter(|w| w[0] == w[1]).count();
    if dups as f64 / (n - 1) as f64 > MAX_DUP_FRACTION { reasons.push(DuplicateFrames); }

    // ---- lip mesh channel ----
    let meshes: Vec<LipMesh> = frames.iter().map(|f| build_lip_mesh(&f.lips_outer, &f.lips_inner)).collect();
    let mouth_w = mean(&meshes.iter().map(|m| m.width_px).collect::<Vec<_>>());
    let open: Vec<f64> = meshes.iter().map(|m| m.openness).collect();
    let spread: Vec<f64> = meshes.iter().map(|m| m.spread).collect();
    let nv = meshes[0].vertices.len();
    let mut max_sd = 0.0f64;
    let cents: Vec<Pt> = meshes.iter().map(|m| centroid(&m.vertices)).collect();
    for k in 0..nv {
        let xs: Vec<f64> = meshes.iter().zip(&cents).map(|(m, c)| (m.vertices.get(k).map_or(0.0, |p| p.0) - c.0) / mouth_w.max(1e-9)).collect();
        let ys: Vec<f64> = meshes.iter().zip(&cents).map(|(m, c)| (m.vertices.get(k).map_or(0.0, |p| p.1) - c.1) / mouth_w.max(1e-9)).collect();
        max_sd = max_sd.max(std_dev(&xs)).max(std_dev(&ys));
    }
    if max_sd < STATIC_MESH_EPS { reasons.push(StaticMesh); }
    let viseme_match = articulation_score(&open, &spread, &t_s);
    if viseme_match < LIPSYNC_ACCEPT { reasons.push(WeakArticulation); }
    let width_factor = ((mouth_w - crate::VISEME_CLIFF_PX) / (crate::MOUTH_HIGH_ACC_PX - crate::VISEME_CLIFF_PX)).clamp(0.0, 1.0);
    ev.lip_llr = 10.0 * (viseme_match - LIPSYNC_ACCEPT) * width_factor.max(0.2);

    // Audio-visual sync (only when the upstream supplies audio visemes).
    if frames.iter().all(|f| f.audio_openness.is_some()) {
        let audio: Vec<f64> = frames.iter().map(|f| f.audio_openness.unwrap()).collect();
        let max_lag = (400.0 / dt_ms).ceil() as i64;
        let (mut best_lag, mut best_c) = (0i64, f64::NEG_INFINITY);
        for lag in -max_lag..=max_lag {
            let c = pearson_lag(&open, &audio, lag);
            if c > best_c { best_c = c; best_lag = lag; }
        }
        let lag_ms = (best_lag as f64 * dt_ms).abs();
        if lag_ms > AV_MAX_LAG_MS || best_c < AV_MIN_CORR { reasons.push(AvDesync); }
        ev.av_llr = (8.0 * (pearson_lag(&open, &audio, 0) - 0.5)).clamp(-8.0, 4.0);
    }

    // ---- glint channel ----
    let mut lock_frac = [0.0; 2];
    let mut ratios = [f64::NAN; 2];
    let mut resid = [f64::NAN; 2];
    let mut light = [(0.0, 0.0); 2];
    let mut eye_ws = Vec::new();
    let (mut rect_frames, mut dbl_frames, mut locked_total) = (0usize, 0usize, 0usize);
    let mut any_motion = false;
    for ei in 0..2usize {
        let (mut di, mut dg, mut off) = (Vec::new(), Vec::new(), Vec::new());
        for f in frames {
            let e = if ei == 0 { &f.left_eye } else { &f.right_eye };
            let w = extent(&e.contour, false).max(1e-9);
            eye_ws.push(w);
            let g = detect_glint(e);
            if let EyeLock::Locked { glint } = g.lock {
                locked_total += 1;
                if g.components >= 2 { dbl_frames += 1; }
                if g.area >= 16 && g.bbox_fill >= 0.95 { rect_frames += 1; }
                let c = centroid(&e.contour);
                di.push(((e.iris_center.0 - c.0) / w, (e.iris_center.1 - c.1) / w));
                dg.push(((glint.0 - c.0) / w, (glint.1 - c.1) / w));
                off.push(((glint.0 - e.iris_center.0) / w, (glint.1 - e.iris_center.1) / w));
            }
        }
        lock_frac[ei] = di.len() as f64 / n as f64;
        if !off.is_empty() { light[ei] = centroid(&off); }
        if lock_frac[ei] < MIN_LOCK_FRACTION || di.len() < 3 { continue; }
        let (mi, mg) = (centroid(&di), centroid(&dg));
        let (mut num, mut den) = (0.0, 0.0);
        for (a, b) in di.iter().zip(dg.iter()) {
            let (ax, ay) = (a.0 - mi.0, a.1 - mi.1);
            num += ax * (b.0 - mg.0) + ay * (b.1 - mg.1);
            den += ax * ax + ay * ay;
        }
        if (den / di.len() as f64).sqrt() < MIN_IRIS_MOTION { continue; }
        any_motion = true;
        let k = num / den;
        ratios[ei] = k.abs();
        let mut e2 = 0.0;
        for (a, b) in di.iter().zip(dg.iter()) {
            let px = mg.0 + k * (a.0 - mi.0) - b.0;
            let py = mg.1 + k * (a.1 - mi.1) - b.1;
            e2 += px * px + py * py;
        }
        resid[ei] = (e2 / di.len() as f64).sqrt();
    }
    let eye_w = mean(&eye_ws);
    let close = eye_w >= CLOSE_EYE_WIDTH_PX;
    let distance_hint = if close { "close" } else { "range" };
    let valid: Vec<f64> = ratios.iter().cloned().filter(|r| r.is_finite()).collect();
    let arc_ratio = if valid.is_empty() { 0.0 } else { mean(&valid) };
    let sigma = glint_sigma(eye_w);
    if lock_frac.iter().all(|&l| l < MIN_LOCK_FRACTION) { reasons.push(GlintNotLocked); }
    else if !any_motion { reasons.push(NoOcularMotion); ev.glint_llr = if close { -3.0 } else { -1.0 }; }
    else {
        ev.glint_llr = glint_llr(arc_ratio, sigma);
        if arc_ratio >= FLAT_GLINT_RATIO { reasons.push(FlatGlintGeometry); }
        if valid.len() == 2 {
            if (valid[0] - valid[1]).abs() <= 3.0 * sigma { ev.binocular_llr = 1.0; }
            else { ev.binocular_llr = -4.0; reasons.push(BinocularInconsistent); }
        }
        let rs: Vec<f64> = resid.iter().cloned().filter(|r| r.is_finite()).collect();
        ev.glint_stability_llr = if mean(&rs) < 0.02 { 1.5 } else { -3.0 };
    }
    if locked_total > 0 {
        if dbl_frames * 2 > locked_total { reasons.push(DoubledGlint); }
        if rect_frames * 2 > locked_total { reasons.push(RectangularGlint); }
    }
    // Same light source → glint sits on the same side of the iris in both eyes.
    let (l0, l1) = (light[0], light[1]);
    let (m0, m1) = ((l0.0 * l0.0 + l0.1 * l0.1).sqrt(), (l1.0 * l1.0 + l1.1 * l1.1).sqrt());
    if m0 > 0.01 && m1 > 0.01 && (l0.0 * l1.0 + l0.1 * l1.1) / (m0 * m1) < LIGHT_MIN_COS {
        reasons.push(GlintLightMismatch);
    }

    // ---- temporal / whole-face liveness ----
    let lm: Vec<Vec<Pt>> = frames.iter().map(all_landmarks).collect();
    let max_res = lm.iter().skip(1).map(|p| affine_residual(&lm[0], p)).fold(0.0, f64::max);
    let moved = lm.iter().any(|p| { let a = centroid(p); let b = centroid(&lm[0]); ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt() > 0.5 });
    if moved && max_res / eye_w.max(1e-9) < PLANAR_RESIDUAL_EPS { reasons.push(PlanarRigidMotion); }

    let aspects: Vec<f64> = frames.iter().map(|f| 0.5 * (eye_aspect(&f.left_eye) + eye_aspect(&f.right_eye))).collect();
    let med_a = median(&aspects);
    let blink = aspects.iter().any(|&a| a < 0.6 * med_a);
    // A 5 s window often contains no blink (normal rate ~8-21/min), so a
    // missing blink is soft evidence only, never a veto on its own.
    ev.blink_llr = if blink { 1.5 } else { reasons.push(NoBlinkObserved); -0.5 };

    let mut jit = Vec::new();
    for k in 2..n {
        let (a, b, c) = (&frames[k - 2], &frames[k - 1], &frames[k]);
        let mut e2 = 0.0; let mut cnt = 0.0;
        for (ea, eb, ec) in [(&a.left_eye, &b.left_eye, &c.left_eye), (&a.right_eye, &b.right_eye, &c.right_eye)] {
            for ((p, q), r) in ea.contour.iter().zip(&eb.contour).zip(&ec.contour) {
                e2 += (p.0 - 2.0 * q.0 + r.0).powi(2) + (p.1 - 2.0 * q.1 + r.1).powi(2); cnt += 1.0;
            }
        }
        if cnt > 0.0 { jit.push((e2 / cnt).sqrt() / eye_w.max(1e-9)); }
    }
    let jitter = median(&jit);
    let static_face = reasons.contains(&StaticMesh) && !any_motion;
    if !static_face && !(JITTER_MIN..=JITTER_MAX).contains(&jitter) { reasons.push(JitterOutOfRange); }
    else if !static_face { ev.jitter_llr = 1.0; }

    // Screen artefacts from the eye crops.
    let moire: Vec<f64> = frames.iter().flat_map(|f| [moire_score(&f.left_eye.patch), moire_score(&f.right_eye.patch)]).flatten().collect();
    if !moire.is_empty() && moire.iter().filter(|&&s| s > MOIRE_ROW_SCORE).count() * 2 > moire.len() { reasons.push(MoirePattern); }
    let bright: Vec<f64> = frames.iter().map(|f| {
        let px = f.left_eye.patch.pixels.iter().chain(f.right_eye.patch.pixels.iter());
        let (s, c) = px.fold((0.0, 0.0), |a, &v| (a.0 + v as f64, a.1 + 1.0));
        if c > 0.0 { s / c } else { 0.0 }
    }).collect();
    let d: Vec<f64> = bright.windows(2).map(|w| w[1] - w[0]).collect();
    let alts = d.windows(2).filter(|w| w[0].abs() > FLICKER_MIN_STEP && w[1].abs() > FLICKER_MIN_STEP && w[0].signum() != w[1].signum()).count();
    if d.len() > 2 && alts as f64 / (d.len() - 1) as f64 > FLICKER_ALT_FRACTION { reasons.push(ScreenFlicker); }

    // ---- fusion ----
    let (w_glint, w_lip) = if close { (1.0, 0.6) } else { (0.3, 1.0) };
    ev.w_glint = w_glint; ev.w_lip = w_lip;
    ev.total_llr = w_glint * (ev.glint_llr + ev.binocular_llr + ev.glint_stability_llr)
        + w_lip * (ev.lip_llr + ev.av_llr) + ev.blink_llr + ev.jitter_llr;
    let posterior = cal.posterior(ev.total_llr);

    let g = GlintReading { arc_ratio, eye_width_px: eye_w, distance_hint: distance_hint.into() };
    let l = LipsyncReading { mouth_width_px: mouth_w, viseme_match };
    let (fused, fused_pass, _) = fuse(&g, &l);

    let vetoes: Vec<ReasonCode> = reasons.iter().cloned().filter(|r| r.is_veto()).collect();
    let ocular_spoof = reasons.contains(&NoOcularMotion) && close;
    let legacy_pass = fused_pass && vetoes.is_empty() && !ocular_spoof;
    let ha_pass = vetoes.is_empty() && !ocular_spoof
        && posterior >= HIGH_ASSURANCE_THRESHOLD && ev.total_llr >= llr_accept_threshold();
    if !fused_pass { reasons.push(FusedBelowThreshold); }
    if !ha_pass { reasons.push(BelowHighAssurance); }
    let passed = match tier { Tier::Legacy => legacy_pass, Tier::HighAssurance => ha_pass };
    if passed { reasons.push(Passed); }
    ScanResult {
        status: if passed { ScanStatus::Pass } else { ScanStatus::Fail }, tier,
        fused_score: fused, legacy_pass, posterior, high_assurance_pass: ha_pass,
        calibrated: cal.calibrated, evidence: ev, glint_score: g.score(), glint_arc_ratio: arc_ratio,
        lipsync_score: l.score(), viseme_match, distance_hint: distance_hint.into(),
        frames_used: n, glint_lock_fraction: lock_frac, vetoes, reasons,
    }
}

/// Synthetic sequence generator shared by the tests and the eval harness.
/// Not a model of real faces: it only exercises each code path.
pub mod synth {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Kind { Live, LiveRange, LiveAudio, PrintedStatic, PrintedMoved, ScreenFlatGlint, ScreenMoire, ScreenFlicker, ScreenRectGlint, ScreenDoubleGlint, ScreenDuplicated, DeepfakeDesync, DeepfakeSmooth, DeepfakeJittery, DeepfakeLight }

    pub const SPOOFS: [Kind; 12] = [Kind::PrintedStatic, Kind::PrintedMoved, Kind::ScreenFlatGlint, Kind::ScreenMoire, Kind::ScreenFlicker, Kind::ScreenRectGlint, Kind::ScreenDoubleGlint, Kind::ScreenDuplicated, Kind::DeepfakeDesync, Kind::DeepfakeSmooth, Kind::DeepfakeJittery, Kind::DeepfakeLight];
    pub const LIVES: [Kind; 3] = [Kind::Live, Kind::LiveRange, Kind::LiveAudio];

    pub struct Rng(pub u64);
    impl Rng { pub fn f(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.0 >> 33) as f64 / (1u64 << 31) as f64) - 0.5 } }

    #[allow(clippy::too_many_arguments)]
    fn eye(cx: f64, cy: f64, s: f64, iris_off: Pt, gr: f64, light: Pt, blink: bool, k: Kind, frame: usize) -> EyeObs {
        let (w, h) = ((90.0 * s) as usize, (50.0 * s) as usize);
        let origin = (cx - 45.0 * s, cy - 25.0 * s);
        let iris = (cx + iris_off.0, cy + iris_off.1);
        let ir = 12.0 * s;
        let glint = (iris.0 + light.0 * s + (gr - 1.0) * iris_off.0, iris.1 + light.1 * s + (gr - 1.0) * iris_off.1);
        let base = if k == Kind::ScreenFlicker { if frame % 2 == 0 { 124.0 } else { 116.0 } } else { 120.0 };
        let mut px = vec![0u8; w * h];
        for y in 0..h { for x in 0..w {
            let (gx, gy) = (x as f64 + origin.0, y as f64 + origin.1);
            let mut v = base;
            if k == Kind::ScreenMoire { v += 25.0 * (gx * std::f64::consts::TAU / 4.0).sin(); }
            if (gx - iris.0).powi(2) + (gy - iris.1).powi(2) <= ir * ir { v = 40.0; }
            let d2 = (gx - glint.0).powi(2) + (gy - glint.1).powi(2);
            match k {
                Kind::ScreenRectGlint => if (gx - glint.0).abs() <= 2.5 && (gy - glint.1).abs() <= 2.5 { v = 255.0; },
                Kind::ScreenDoubleGlint => {
                    if d2 <= 2.25 { v = 255.0 - 40.0 * d2; }
                    let d3 = (gx - glint.0 - 6.0).powi(2) + (gy - glint.1).powi(2);
                    if d3 <= 2.25 { v = 255.0 - 40.0 * d3; }
                }
                _ => if d2 <= 2.25 * s * s { v = 255.0 - 40.0 * d2 / (s * s); },
            }
            px[y * w + x] = v.clamp(0.0, 255.0) as u8;
        }}
        let ry = if blink { 3.0 } else { 12.0 } * s;
        let contour = (0..16).map(|q| { let a = q as f64 / 16.0 * std::f64::consts::TAU; (cx + 35.0 * s * a.cos(), cy + ry * a.sin()) }).collect();
        EyeObs { contour, iris_center: iris, iris_radius: ir, patch: EyePatch { origin, width: w, height: h, pixels: px } }
    }

    fn lips(cx: f64, cy: f64, s: f64, open: f64) -> (Vec<Pt>, Vec<Pt>) {
        let o = (0..20).map(|q| { let a = q as f64 / 20.0 * std::f64::consts::TAU; (cx + 60.0 * s * a.cos(), cy + (14.0 + 20.0 * open) * s * a.sin()) }).collect();
        let i = (0..20).map(|q| { let a = q as f64 / 20.0 * std::f64::consts::TAU; (cx + (45.0 - 4.0 * open) * s * a.cos(), cy + (1.0 + 18.0 * open) * s * a.sin()) }).collect();
        (o, i)
    }

    pub fn sequence(k: Kind, n: usize, dt: u64, seed: u64) -> Vec<FrameObs> {
        use Kind::*;
        let mut r = Rng(seed);
        let phase = (seed % 7) as f64 * 0.37;
        let s = if k == LiveRange { 0.6 } else { 1.0 };
        let printed = matches!(k, PrintedStatic | PrintedMoved);
        let jit = match k { DeepfakeSmooth | PrintedStatic | PrintedMoved => 0.0, DeepfakeJittery => 14.0, _ => 1.0 };
        let gr = match k { ScreenFlatGlint | PrintedStatic | PrintedMoved => 1.0, _ => 0.5 };
        let raw: Vec<FrameObs> = (0..n).map(|i| {
            let t = i as f64 * dt as f64 / 1000.0;
            let (hx, hy) = if k == PrintedStatic { (0.0, 0.0) } else { (3.0 * (0.4 * t + phase).sin() + jit * r.f(), 2.0 * (0.3 * t).cos() + jit * r.f()) };
            let io = if printed { (0.0, 0.0) } else { (6.0 * s * (2.0 * t + phase).sin(), 2.5 * s * (1.3 * t).cos()) };
            let open_f = |t: f64| 0.5 + 0.3 * (std::f64::consts::TAU * 3.0 * t + phase).sin() + 0.15 * (std::f64::consts::TAU * 1.13 * t).sin();
            let open = if printed { 0.3 } else { open_f(t) + 0.03 * r.f() };
            let blink = !printed && k != DeepfakeSmooth && (2.0..2.15).contains(&t);
            let le = if k == DeepfakeLight { (-2.0, -2.0) } else { (2.0, -2.0) };
            // at range the mouth is the dominant cue (mirrors the legacy range fixture: small eyes, wide mouth)
            let ls = if k == LiveRange { 1.7 } else { 1.0 };
            let (lo, li) = lips(320.0 + hx, 400.0 + hy, ls, open);
            let mut f = FrameObs {
                t_ms: i as u64 * dt,
                left_eye: eye(250.0 + hx, 250.0 + hy, s, io, gr, le, blink, k, i),
                right_eye: eye(250.0 + 140.0 * s + hx, 250.0 + hy, s, io, gr, (2.0, -2.0), blink, k, i),
                lips_outer: lo, lips_inner: li,
                audio_openness: match k { LiveAudio => Some(open_f(t)), DeepfakeDesync => Some(open_f(t - 0.25 + 0.0)), _ => None },
            };
            if k == PrintedMoved {
                // rotate + scale the whole printed face about its centre (planar motion)
                let (ang, sc) = (0.05 * (0.8 * t).sin(), 1.0 + 0.03 * (0.5 * t).sin());
                let (c0, s0) = (ang.cos() * sc, ang.sin() * sc);
                let tr = |p: &mut Pt| { let (x, y) = (p.0 - 320.0, p.1 - 320.0); *p = (320.0 + c0 * x - s0 * y, 320.0 + s0 * x + c0 * y); };
                for e in [&mut f.left_eye, &mut f.right_eye] { e.contour.iter_mut().for_each(tr); tr(&mut e.iris_center); }
                f.lips_outer.iter_mut().for_each(tr); f.lips_inner.iter_mut().for_each(tr);
            }
            f
        }).collect();
        if k == ScreenDuplicated {
            // a 15 fps replay up-sampled to 30 fps: every frame shown twice
            raw.iter().enumerate().map(|(i, f)| { let mut g = raw[i / 2].clone(); g.t_ms = f.t_ms; g }).collect()
        } else { raw }
    }
}

#[cfg(test)]
mod tests {
    use super::synth::{sequence, Kind, LIVES, SPOOFS};
    use super::*;

    fn run(k: Kind, seed: u64) -> ScanResult { scan_window(true, sequence(k, 150, 33, seed)) }

    #[test]
    fn live_sequence_passes_high_assurance() {
        let r = run(Kind::Live, 7);
        assert_eq!(r.status, ScanStatus::Pass, "{:?}", r);
        assert!(r.high_assurance_pass && r.legacy_pass);
        assert!(r.posterior >= 0.999 && r.evidence.total_llr >= llr_accept_threshold(), "{:?}", r.evidence);
        assert!((r.glint_arc_ratio - 0.5).abs() < 0.12);
        assert!(!r.calibrated);
    }

    #[test]
    fn live_with_audio_passes() {
        let r = run(Kind::LiveAudio, 3);
        assert_eq!(r.status, ScanStatus::Pass, "{:?}", r);
        assert!(r.evidence.av_llr > 0.0);
    }

    #[test]
    fn live_at_range_passes_legacy_only() {
        // Glint pixels collapse at range; without audio the evidence stays
        // below the high-assurance bound by design.
        let r = run(Kind::LiveRange, 4);
        assert_eq!(r.distance_hint, "range");
        assert!(r.vetoes.is_empty(), "{:?}", r);
        assert!(!r.high_assurance_pass);
        let r2 = scan_window_with(true, sequence(Kind::LiveRange, 150, 33, 4), Calibration::default(), Tier::Legacy);
        assert!(r2.legacy_pass, "{:?}", r2);
        assert_eq!(r2.status, ScanStatus::Pass);
    }

    fn expect_veto(k: Kind, code: ReasonCode) {
        let r = run(k, 11);
        assert_eq!(r.status, ScanStatus::Fail, "{:?} {:?}", k, r);
        assert!(r.vetoes.contains(&code), "{:?} expected {:?}, got {:?}", k, code, r.reasons);
        assert!(!r.legacy_pass && !r.high_assurance_pass);
    }

    #[test] fn static_photo_fails() { expect_veto(Kind::PrintedStatic, ReasonCode::StaticMesh); }
    #[test] fn moved_photo_planar_motion() { expect_veto(Kind::PrintedMoved, ReasonCode::PlanarRigidMotion); }
    #[test] fn flat_screen_glint_fails() { expect_veto(Kind::ScreenFlatGlint, ReasonCode::FlatGlintGeometry); }
    #[test] fn moire_fails() { expect_veto(Kind::ScreenMoire, ReasonCode::MoirePattern); }
    #[test] fn flicker_fails() { expect_veto(Kind::ScreenFlicker, ReasonCode::ScreenFlicker); }
    #[test] fn rect_glint_fails() { expect_veto(Kind::ScreenRectGlint, ReasonCode::RectangularGlint); }
    #[test] fn double_glint_fails() { expect_veto(Kind::ScreenDoubleGlint, ReasonCode::DoubledGlint); }
    #[test] fn duplicated_frames_fail() { expect_veto(Kind::ScreenDuplicated, ReasonCode::DuplicateFrames); }
    #[test] fn av_desync_fails() { expect_veto(Kind::DeepfakeDesync, ReasonCode::AvDesync); }
    #[test] fn smooth_deepfake_fails() { expect_veto(Kind::DeepfakeSmooth, ReasonCode::JitterOutOfRange); }
    #[test] fn jittery_deepfake_fails() { expect_veto(Kind::DeepfakeJittery, ReasonCode::JitterOutOfRange); }
    #[test] fn light_mismatch_fails() { expect_veto(Kind::DeepfakeLight, ReasonCode::GlintLightMismatch); }

    #[test]
    fn too_few_frames_inconclusive() {
        let r = scan_window(true, sequence(Kind::Live, 10, 33, 1));
        assert_eq!(r.status, ScanStatus::Inconclusive);
        assert_eq!(r.reasons, vec![ReasonCode::TooFewFrames]);
    }

    #[test]
    fn window_timing_respected() {
        let r = scan_window(true, sequence(Kind::Live, 300, 33, 5));
        assert!(r.frames_used <= (WINDOW_MS / 33 + 1) as usize, "{}", r.frames_used);
        let r = scan_window(true, sequence(Kind::Live, 60, 16, 5));
        assert_eq!(r.reasons, vec![ReasonCode::WindowTooShort]);
    }

    #[test]
    fn non_monotonic_timestamps_fail() {
        let mut f = sequence(Kind::Live, 100, 33, 2);
        f[50].t_ms = 10;
        let r = scan_window(true, f);
        assert_eq!(r.reasons, vec![ReasonCode::TimestampNonMonotonic]);
        assert_eq!(r.status, ScanStatus::Fail);
    }

    #[test]
    fn no_consent_no_scan() {
        assert!(SilentScanner::new(false).is_none());
        let r = scan_window(false, sequence(Kind::Live, 150, 33, 7));
        assert_eq!(r.reasons, vec![ReasonCode::NoConsent]);
        assert_eq!(r.frames_used, 0);
    }

    #[test]
    fn glint_lock_and_mesh() {
        let f = &sequence(Kind::Live, 1, 33, 1)[0];
        assert!(matches!(find_glint(&f.left_eye), EyeLock::Locked { .. }));
        let m = build_lip_mesh(&f.lips_outer, &f.lips_inner);
        assert_eq!(m.triangles.len(), 40);
        assert!((m.width_px - 120.0).abs() < 1e-6 && m.openness > 0.0);
    }

    #[test]
    fn glint_llr_shape() {
        let s = glint_sigma(70.0);
        assert!(glint_llr(0.5, s) > 11.0);
        assert!(glint_llr(1.0, s) < -11.0);
        assert!(glint_llr(0.75, s).abs() < 1e-9);
        assert!((llr_accept_threshold() - 11.5129).abs() < 1e-3);
    }

    #[test]
    fn calibration_config_parses() {
        let c = Calibration::from_config("# fitted\nplatt_a=0.8\nplatt_b=-1.5\ncalibrated=true\nisotonic=0.0:0.0,0.5:0.4,1.0:1.0\n").unwrap();
        assert!(c.calibrated && (c.platt_a - 0.8).abs() < 1e-12 && c.isotonic.len() == 3);
        let d = Calibration::default();
        assert!(!d.calibrated);
        assert!((d.posterior(0.0) - 0.5).abs() < 1e-12);
        assert!(Calibration::from_config("garbage").is_err());
    }

    /// Evaluation harness on SYNTHETIC data only: reports empirical FAR/FRR.
    /// These numbers say nothing about real-world accuracy.
    #[test]
    fn eval_harness_synthetic() {
        let seeds = 0..8u64;
        let (mut fa, mut spoofs) = (0, 0);
        for k in SPOOFS { for s in seeds.clone() { spoofs += 1; let r = run(k, s); if r.status == ScanStatus::Pass || r.legacy_pass { fa += 1; eprintln!("FALSE ACCEPT {:?} seed {} {:?}", k, s, r.reasons); } } }
        let (mut fr, mut lives) = (0, 0);
        for k in [Kind::Live, Kind::LiveAudio] { for s in seeds.clone() { lives += 1; if run(k, s).status != ScanStatus::Pass { fr += 1; } } }
        let mut fr_legacy_range = 0;
        for s in seeds.clone() { if !run(Kind::LiveRange, s).legacy_pass { fr_legacy_range += 1; } }
        eprintln!("synthetic eval: FAR {}/{} , FRR(high-assurance, close) {}/{} , FRR(legacy, range) {}/8 , live kinds {:?}", fa, spoofs, fr, lives, fr_legacy_range, LIVES);
        assert_eq!(fa, 0);
        assert_eq!(fr, 0);
        assert_eq!(fr_legacy_range, 0);
    }
}
