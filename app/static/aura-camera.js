// AURA live camera adapter: MediaPipe Tasks Vision FaceLandmarker → /api/aura/scan frame JSON.
// No overlay is drawn on the face. Eye crops are grayscale, in memory only, sent once with the
// scan request and discarded by the server after scoring. All camera tracks stop when the window ends.
const MP_VER = "0.10.14";
const MP_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/vision_bundle.mjs`;
const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/wasm`;
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
// Same indices as src/aura_scan.py capture_and_scan (MediaPipe 478-point mesh).
const L_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const R_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
const L_IRIS = [468, 469, 470, 471, 472], R_IRIS = [473, 474, 475, 476, 477];
const LIPS_O = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
const LIPS_I = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];
const MESH_PTS = [1, 168, 33, 263, 133, 362, 61, 291, 199, 10, 234, 454]; // nose tip/bridge, eye corners, mouth corners, chin, forehead, cheek edges
const BLEND_KEEP = ["jawOpen", "mouthPucker", "mouthFunnel", "mouthClose", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight",
  "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight"];

let landmarker = null, loadErr = null;
export async function load() {
  if (landmarker) return landmarker;
  try {
    const { FaceLandmarker, FilesetResolver } = await import(MP_URL);
    const fs = await FilesetResolver.forVisionTasks(WASM);
    const opts = (delegate) => ({ baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: "VIDEO", numFaces: 1,
      outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
    try { landmarker = await FaceLandmarker.createFromOptions(fs, opts("GPU")); }
    catch { landmarker = await FaceLandmarker.createFromOptions(fs, opts("CPU")); }
    return landmarker;
  } catch (e) { loadErr = e; throw new Error("MediaPipe failed to load (offline?): " + (e.message || e)); }
}

// Yaw/pitch (degrees) from MediaPipe's column-major 4x4 facial transformation matrix.
function pose(m) {
  if (!m) return null;
  const d = m.data; // column-major: r[row][col] = d[col*4+row]
  const r20 = d[2], r21 = d[6], r22 = d[10];
  return { yaw: Math.atan2(-r20, Math.hypot(r21, r22)) * 180 / Math.PI, pitch: Math.atan2(r21, r22) * 180 / Math.PI };
}

/** Capture `durationMs` of webcam frames. Requests permission (browser prompt is never suppressed). */
// Screen-flash colour challenge overlay. Photosensitivity: >= 450 ms per colour (<= 2.2 transitions/s),
// 100 ms fades, 55% opacity of moderate colours, no white/black strobing. Timed on the frames' t0 clock.
export function runFlash(flash, t0) {
  if (!flash || !flash.seq || !flash.seq.length) return () => {};
  const d = document.createElement("div"), fade = flash.fade_ms || 100, timers = [];
  d.style.cssText = `position:fixed;inset:0;z-index:9999;pointer-events:none;opacity:0;transition:background-color ${fade}ms linear,opacity ${fade}ms linear;display:flex;align-items:flex-end;justify-content:center;font:600 14px system-ui;color:#fff;padding-bottom:24px`;
  d.textContent = "Colour check · keep facing the screen";
  document.body.appendChild(d);
  const at = (ms, fn) => timers.push(setTimeout(fn, Math.max(0, ms - (performance.now() - t0))));
  const el = performance.now() - t0; // segments already past are skipped (never burst-fire)
  for (const s of flash.seq) if (s.start_ms + s.dur_ms > el + fade) at(s.start_ms, () => { d.style.backgroundColor = `rgb(${s.rgb.join(",")})`; d.style.opacity = "0.55"; });
  const last = flash.seq[flash.seq.length - 1], end = last.start_ms + last.dur_ms;
  at(end, () => { d.style.opacity = "0"; });
  const kill = () => { timers.forEach(clearTimeout); d.remove(); };
  at(end + fade + 50, kill);
  return kill;
}

export async function capture(durationMs, { t0 = performance.now(), onStatus = () => {}, mic = false, flash = null } = {}) {
  const lm = await load();
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: !!mic });
  const video = document.createElement("video");
  video.muted = true; video.playsInline = true; video.srcObject = stream;
  await video.play();
  const W = video.videoWidth, H = video.videoHeight;
  const cv = new OffscreenCanvas(W, H), cx = cv.getContext("2d", { willReadFrequently: true });
  let audio = null;
  if (mic && stream.getAudioTracks().length) {
    const ac = new AudioContext(), an = ac.createAnalyser(); an.fftSize = 1024;
    ac.createMediaStreamSource(stream).connect(an);
    const buf = new Float32Array(an.fftSize);
    audio = { ac, read() { an.getFloatTimeDomainData(buf); let e = 0; for (const v of buf) e += v * v; return Math.sqrt(e / buf.length); } };
  }
  // PAD crops: <= 5 face crops (80x80 BGR, 2.7x face box), ~1 s apart, sent once with the scan, never stored.
  const padCv = new OffscreenCanvas(80, 80), padCx = padCv.getContext("2d", { willReadFrequently: true });
  const padCrops = []; let padNext = 500;
  const frames = [], stats = { seen: 0, faces: 0, width: W, height: H, yaw: [] };
  const killFlash = runFlash(flash, t0); stats.flash = !!flash;
  let lastTime = -1;
  try {
    onStatus(`Camera active — liveness check in progress (${W}×${H})`);
    const end = performance.now() + durationMs;
    while (performance.now() < end) {
      await new Promise((res) => (video.requestVideoFrameCallback ? video.requestVideoFrameCallback(() => res()) : requestAnimationFrame(() => res())));
      if (video.currentTime === lastTime) continue; // only new video frames
      lastTime = video.currentTime; stats.seen++;
      const now = performance.now();
      const r = lm.detectForVideo(video, now);
      if (!r.faceLandmarks || !r.faceLandmarks.length) continue;
      stats.faces++;
      const L = r.faceLandmarks[0];
      const P = (i) => [L[i].x * W, L[i].y * H]; // normalized → pixels
      cx.drawImage(video, 0, 0, W, H);
      const eye = (cids, iids) => {
        const cont = cids.map(P), iris = iids.map(P), ctr = iris[0];
        const rad = iris.slice(1).reduce((a, p) => a + Math.hypot(p[0] - ctr[0], p[1] - ctr[1]), 0) / 4;
        let x0 = Math.floor(Math.min(...cont.map((p) => p[0]))) - 4, x1 = Math.floor(Math.max(...cont.map((p) => p[0]))) + 4;
        let y0 = Math.floor(Math.min(...cont.map((p) => p[1]))) - 8, y1 = Math.floor(Math.max(...cont.map((p) => p[1]))) + 8;
        x0 = Math.max(x0, 0); y0 = Math.max(y0, 0); x1 = Math.min(x1, W); y1 = Math.min(y1, H);
        const w = Math.max(x1 - x0, 1), h = Math.max(y1 - y0, 1);
        const rgba = cx.getImageData(x0, y0, w, h).data, px = new Array(w * h);
        for (let i = 0, j = 0; j < px.length; i += 4, j++) px[j] = Math.round(0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]); // = cv2 BGR2GRAY
        return { contour: cont, iris_center: ctr, iris_radius: rad, patch: { origin: [x0, y0], width: w, height: h, pixels: px } };
      };
      const f = { t_ms: Math.round(now - t0), left_eye: eye(L_EYE, L_IRIS), right_eye: eye(R_EYE, R_IRIS),
        lips_outer: LIPS_O.map(P), lips_inner: LIPS_I.map(P) };
      // Canonical-frame extras (expression coefficients + pose; not identity, not stored).
      const z = (i) => L[i].z * W;
      f.lip_z = [z(0), z(17), z(13), z(14), z(61), z(291)];
      f.mesh2d = MESH_PTS.map(P); // 12 sparse pixel landmarks for the planar-vs-3D (parallax) check
      // rPPG scalars only: mean R,G,B of 3 skin ROIs (forehead, cheeks), box half-size 0.15*IOD. No pixels leave.
      { const iod = Math.hypot(P(468)[0] - P(473)[0], P(468)[1] - P(473)[1]), hs = Math.max(2, Math.round(0.15 * iod));
        const roi = (c) => { const x0 = Math.max(0, Math.round(c[0] - hs)), y0 = Math.max(0, Math.round(c[1] - hs));
          const w = Math.max(1, Math.min(W - x0, 2 * hs)), h = Math.max(1, Math.min(H - y0, 2 * hs)), d = cx.getImageData(x0, y0, w, h).data;
          let r = 0, g = 0, b = 0; for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
          const n = d.length / 4; return [+(r / n).toFixed(3), +(g / n).toFixed(3), +(b / n).toFixed(3)]; };
        const fh = [(P(10)[0] + P(151)[0]) / 2, (P(10)[1] + P(151)[1]) / 2];
        f.rppg = { forehead: roi(fh), lcheek: roi(P(50)), rcheek: roi(P(280)), bg: roi([hs + 2, hs + 2]) }; }
      f.iod_px = Math.hypot(P(468)[0] - P(473)[0], P(468)[1] - P(473)[1]);
      const ps = pose(r.facialTransformationMatrixes && r.facialTransformationMatrixes[0]);
      if (ps) { f.yaw = ps.yaw; f.pitch = ps.pitch; stats.yaw.push(ps.yaw); }
      if (r.faceBlendshapes && r.faceBlendshapes[0]) {
        f.blend = {}; for (const c of r.faceBlendshapes[0].categories) if (BLEND_KEEP.includes(c.categoryName)) f.blend[c.categoryName] = +c.score.toFixed(4);
      }
      if (padCrops.length < 5 && f.t_ms >= padNext) {
        padNext = f.t_ms + 1000;
        let x0 = W, y0 = H, x1 = 0, y1 = 0; for (const q of L) { x0 = Math.min(x0, q.x * W); x1 = Math.max(x1, q.x * W); y0 = Math.min(y0, q.y * H); y1 = Math.max(y1, q.y * H); }
        const bw = x1 - x0, bh = y1 - y0, sc = Math.min(2.7, (H - 1) / bh, (W - 1) / bw), cxm = (x0 + x1) / 2, cym = (y0 + y1) / 2;
        const sx = Math.max(0, cxm - bw * sc / 2), sy = Math.max(0, cym - bh * sc / 2), sw = Math.min(W - sx, bw * sc), sh = Math.min(H - sy, bh * sc);
        padCx.drawImage(cv, sx, sy, sw, sh, 0, 0, 80, 80);
        const d = padCx.getImageData(0, 0, 80, 80).data, px = new Array(80 * 80 * 3);
        for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { px[j] = d[i + 2]; px[j + 1] = d[i + 1]; px[j + 2] = d[i]; } // RGBA -> BGR
        padCrops.push(px);
      }
      if (audio) f.audio_level = audio.read();
      frames.push(f);
    }
  } finally {
    killFlash(); stream.getTracks().forEach((t) => t.stop()); // stop camera (and mic) right after the window
    video.srcObject = null; cx.clearRect(0, 0, W, H); padCx.clearRect(0, 0, 80, 80);
    if (audio) audio.ac.close();
  }
  if (audio && frames.length) { // normalize energy envelope to 0..1 → audio_openness (A/V sync channel)
    const mx = Math.max(...frames.map((f) => f.audio_level), 1e-9);
    frames.forEach((f) => { f.audio_openness = f.audio_level / mx; delete f.audio_level; });
  }
  onStatus(`Camera stopped · ${stats.faces}/${stats.seen} frames had a face`);
  stats.pad_crops = padCrops.length;
  return { frames, stats, padCrops };
}
