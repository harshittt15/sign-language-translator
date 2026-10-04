/* Real-Time Sign Language Translator, browser client.
 *
 * MediaPipe HandLandmarker runs HERE, in the browser. Only the 21 raw
 * landmarks are posted to the backend, which normalizes them, scales them and
 * classifies them with the Random Forest. Nothing about the feature
 * construction happens in JavaScript: Python remains the single source of
 * truth for wrist-centering, scale normalization and the 63 feature vector.
 *
 * Two coordinate rules matter and are easy to get wrong:
 *   1. Detection runs on the centred SQUARE region of the video, matching the
 *      server's center_square_crop(). A 16:9 frame would stretch x relative to
 *      y and displace every feature.
 *   2. The frame fed to the detector is NEVER mirrored. The preview is flipped
 *      in CSS only, for a natural self view.
 */
import {
  FilesetResolver,
  HandLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18';

const WASM_BASE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
// Served by our own backend so the browser uses the identical model bundle the
// training features were extracted with.
const MODEL_URL = '/model/hand_landmarker.task';

const MIN_POST_INTERVAL = 40;     // ms; detection runs faster than we post

const $ = (id) => document.getElementById(id);
const els = {
  video: $('video'), overlay: $('overlay'), cropGuide: $('cropGuide'),
  stateIdle: $('stateIdle'), stateBusy: $('stateBusy'), stateError: $('stateError'),
  errTitle: $('errTitle'), errText: $('errText'),
  startBtn: $('startBtn'), retryBtn: $('retryBtn'), stopBtn: $('stopBtn'),
  audioBtn: $('audioBtn'), clearBtn: $('clearBtn'),
  glyph: $('glyph'), rawLine: $('rawLine'),
  confText: $('confText'), confFill: $('confFill'),
  transcript: $('transcript'), charCount: $('charCount'),
  dotCam: $('dotCam'), txtCam: $('txtCam'),
  dotHand: $('dotHand'), txtHand: $('txtHand'),
  dotApi: $('dotApi'), txtApi: $('txtApi'),
  fpsItem: $('fpsItem'), txtFps: $('txtFps'),
};

const sessionId = (crypto.randomUUID && crypto.randomUUID()) ||
                  String(Date.now()) + Math.random().toString(16).slice(2);

let stream = null;
let landmarker = null;
let running = false;
let inFlight = false;
let audioOn = false;
let transcript = '';
let lastPost = 0;
let lastVideoTime = -1;
const squareCanvas = document.createElement('canvas');
const rtt = [];

/* ---------------- view state ---------------- */

function showState(name) {
  els.stateIdle.hidden  = name !== 'idle';
  els.stateBusy.hidden  = name !== 'busy';
  els.stateError.hidden = name !== 'error';
  els.cropGuide.hidden  = name !== 'live';
}

const setDot = (dot, cls) => {
  dot.classList.remove('is-on', 'is-off', 'is-wait');
  if (cls) dot.classList.add(cls);
};
const setCamera = (cls, label) => { setDot(els.dotCam, cls); els.txtCam.textContent = label; };
const setHand = (ok) => {
  setDot(els.dotHand, ok ? 'is-on' : null);
  els.txtHand.textContent = ok ? 'Hand detected' : 'No hand';
};
const setApi = (cls, label) => { setDot(els.dotApi, cls); els.txtApi.textContent = label; };

/* ---------------- backend ---------------- */

let settings = {
  min_hand_detection_confidence: 0.7,
  min_tracking_confidence: 0.5,
  num_hands: 2,
};

async function checkHealth() {
  try {
    const b = await (await fetch('/health')).json();
    setApi(b.status === 'ok' && b.model_loaded ? 'is-on' : 'is-off',
           b.status === 'ok' && b.model_loaded ? 'Backend ready' : 'Model not loaded');
  } catch {
    setApi('is-off', 'Backend unreachable');
  }
}

async function loadSettings() {
  try {
    const b = await (await fetch('/model-info')).json();
    // Mirror the server's MediaPipe thresholds so the landmarks the browser
    // produces match what the model was trained on.
    settings = {
      min_hand_detection_confidence: b.min_hand_detection_confidence,
      min_tracking_confidence: b.min_tracking_confidence,
      num_hands: b.num_hands,
    };
  } catch { /* fall back to the defaults above */ }
}

/* ---------------- landmarker ---------------- */

async function initLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numHands: settings.num_hands,
    minHandDetectionConfidence: settings.min_hand_detection_confidence,
    minHandPresenceConfidence: settings.min_tracking_confidence,
    minTrackingConfidence: settings.min_tracking_confidence,
  });
}

/* ---------------- camera ---------------- */

async function waitForMetadata() {
  if (els.video.readyState >= 1 && els.video.videoWidth > 0) return;
  await new Promise((resolve) => {
    let done = false;
    const fin = () => {
      if (done) return;
      done = true;
      els.video.removeEventListener('loadedmetadata', fin);
      clearTimeout(t);
      resolve();
    };
    const t = setTimeout(fin, 4000);
    els.video.addEventListener('loadedmetadata', fin);
  });
}

async function startCamera() {
  if (running) return;
  showState('busy');
  setCamera('is-wait', 'Requesting access');
  els.startBtn.disabled = true;
  els.retryBtn.disabled = true;

  let s;
  try {
    s = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
  } catch (err) { failCamera(err); return; }

  try {
    stream = s;
    els.video.srcObject = s;
    await waitForMetadata();
    try { await els.video.play(); } catch { /* may already be playing */ }

    if (!landmarker) {
      els.errTitle.textContent = 'Loading hand model';
      await initLandmarker();
    }
    goLive();
  } catch (err) { failCamera(err); }
}

function goLive() {
  running = true;
  lastVideoTime = -1;
  showState('live');
  setCamera('is-on', 'Camera on');
  setHand(false);
  els.stopBtn.disabled = false;
  els.startBtn.disabled = false;
  els.retryBtn.disabled = false;
  els.fpsItem.hidden = false;
  requestAnimationFrame(loop);
}

function failCamera(err) {
  running = false;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  els.video.srcObject = null;

  const n = err && err.name;
  let title = 'Camera unavailable';
  let text = 'Something stopped the camera from starting. Please try again.';
  if (n === 'NotAllowedError' || n === 'SecurityError') {
    title = 'Camera permission denied';
    text = 'Your browser blocked camera access. Allow it from the permissions ' +
           'menu in the address bar, then try again.';
  } else if (n === 'NotFoundError' || n === 'OverconstrainedError') {
    title = 'No camera found';
    text = 'No webcam was detected. Connect one and try again.';
  } else if (n === 'NotReadableError' || n === 'AbortError') {
    title = 'Camera already in use';
    text = 'Another application is using the webcam. Close it, then try again.';
  } else if (err instanceof Error) {
    title = 'Could not start hand tracking';
    text = 'The hand landmark model failed to load. Check your connection and ' +
           'try again.';
  }
  els.errTitle.textContent = title;
  els.errText.textContent = text;
  showState('error');
  setCamera('is-off', 'Camera blocked');
  setHand(false);
  els.stopBtn.disabled = true;
  els.startBtn.disabled = false;
  els.retryBtn.disabled = false;
  els.fpsItem.hidden = true;
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  els.video.srcObject = null;
  showState('idle');
  setCamera(null, 'Camera off');
  setHand(false);
  els.stopBtn.disabled = true;
  els.startBtn.disabled = false;
  els.retryBtn.disabled = false;
  els.fpsItem.hidden = true;
  els.txtFps.textContent = '';
  rtt.length = 0;
  showPrediction(null, null, 0);
  clearOverlay();
}

/* ---------------- detect and post ---------------- */

function loop() {
  if (!running) return;
  const v = els.video;

  if (landmarker && v.videoWidth > 0 && v.currentTime !== lastVideoTime) {
    lastVideoTime = v.currentTime;

    // Centred square, matching the server's center_square_crop(). Drawn
    // UNMIRRORED: the preview flip is CSS only.
    const side = Math.min(v.videoWidth, v.videoHeight);
    const sx = (v.videoWidth - side) / 2;
    const sy = (v.videoHeight - side) / 2;
    if (squareCanvas.width !== side) {
      squareCanvas.width = side;
      squareCanvas.height = side;
    }
    squareCanvas.getContext('2d')
      .drawImage(v, sx, sy, side, side, 0, 0, side, side);

    let hand = null;
    try {
      const res = landmarker.detectForVideo(squareCanvas, performance.now());
      if (res && res.landmarks && res.landmarks.length) hand = res.landmarks[0];
    } catch { /* a dropped frame is not fatal */ }

    drawLandmarks(hand);
    setHand(!!hand);

    const now = performance.now();
    if (!inFlight && now - lastPost >= MIN_POST_INTERVAL) {
      lastPost = now;
      postLandmarks(hand);
    }
  }
  requestAnimationFrame(loop);
}

async function postLandmarks(hand) {
  inFlight = true;
  const t0 = performance.now();
  try {
    const body = {
      session_id: sessionId,
      // null tells the server no hand was seen, which advances the
      // smoother's no-hand counter instead of silently stalling it.
      landmarks: hand
        ? hand.map((p) => ({ x: p.x, y: p.y, z: p.z ?? 0 }))
        : null,
    };
    const resp = await fetch('/predict-landmarks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const out = await resp.json();

    setApi('is-on', 'Backend ready');
    handleResult(out);
    rtt.push(performance.now() - t0);
    if (rtt.length > 20) rtt.shift();
    const avg = rtt.reduce((a, b) => a + b, 0) / rtt.length;
    els.txtFps.textContent = Math.round(avg) + ' ms';
  } catch {
    setApi('is-off', 'Connection lost');
  } finally {
    inFlight = false;
  }
}

/* ---------------- rendering ---------------- */

function handleResult(out) {
  showPrediction(out.sign, out.raw_sign, out.confidence || 0);
  if (out.emitted && out.sign) {
    transcript += out.sign;
    els.transcript.textContent = transcript;
    els.charCount.textContent =
      transcript.length + (transcript.length === 1 ? ' character' : ' characters');
    speak(out.sign);
  }
}

function showPrediction(sign, raw, conf) {
  els.glyph.textContent = sign || ' ';
  els.glyph.classList.toggle('is-idle', !sign);
  els.rawLine.textContent = (raw && raw !== sign) ? 'this frame: ' + raw : '';
  const pct = Math.round(conf * 100);
  els.confText.textContent = pct + '%';
  els.confFill.style.width = pct + '%';
  els.confFill.classList.toggle('is-low', conf > 0 && conf < 0.6);
}

function clearOverlay() {
  const c = els.overlay;
  c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

const BONES = [
  [0,1],[1,2],[2,3],[3,4], [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12], [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20], [5,9],[9,13],[13,17],
];

function drawLandmarks(hand) {
  const c = els.overlay;
  const rect = c.getBoundingClientRect();
  if (c.width !== Math.round(rect.width) || c.height !== Math.round(rect.height)) {
    c.width = Math.round(rect.width);
    c.height = Math.round(rect.height);
  }
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (!hand) return;

  const vw = els.video.videoWidth, vh = els.video.videoHeight;
  if (!vw || !vh) return;

  // Landmarks are [0,1] inside the square crop. Map square to video pixels,
  // then apply the CSS object-fit cover transform and the mirrored preview.
  const side = Math.min(vw, vh);
  const cropX = (vw - side) / 2, cropY = (vh - side) / 2;
  const scale = Math.max(c.width / vw, c.height / vh);
  const ox = (c.width - vw * scale) / 2;
  const oy = (c.height - vh * scale) / 2;

  const pts = hand.map((p) => [
    c.width - (ox + (cropX + p.x * side) * scale),
    oy + (cropY + p.y * side) * scale,
  ]);

  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(79,193,166,.85)';
  ctx.beginPath();
  for (const [a, b] of BONES) {
    if (!pts[a] || !pts[b]) continue;
    ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]);
  }
  ctx.stroke();

  ctx.fillStyle = '#f2f3f5';
  for (const [x, y] of pts) {
    ctx.beginPath();
    ctx.arc(x, y, 2.8, 0, Math.PI * 2);
    ctx.fill();
  }
}

/* ---------------- audio, clear ---------------- */

function speak(text) {
  if (!audioOn || !('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;
  window.speechSynthesis.speak(u);
}

function toggleAudio() {
  audioOn = !audioOn;
  els.audioBtn.setAttribute('aria-pressed', String(audioOn));
  els.audioBtn.textContent = audioOn ? 'Speaking letters' : 'Speak letters';
  if (!audioOn && 'speechSynthesis' in window) window.speechSynthesis.cancel();
}

async function clearText() {
  transcript = '';
  els.transcript.textContent = '';
  els.charCount.textContent = '';
  try {
    const fd = new FormData();
    fd.append('session_id', sessionId);
    await fetch('/reset', { method: 'POST', body: fd });
  } catch { /* clearing locally is still correct */ }
}

/* ---------------- startup ---------------- */

els.startBtn.addEventListener('click', startCamera);
els.retryBtn.addEventListener('click', startCamera);
els.stopBtn.addEventListener('click', stopCamera);
els.audioBtn.addEventListener('click', toggleAudio);
els.clearBtn.addEventListener('click', clearText);
window.addEventListener('beforeunload', () => { if (stream) stopCamera(); });
els.video.addEventListener('emptied', () => { if (running) stopCamera(); });

async function init() {
  showState('idle');
  setCamera(null, 'Camera off');
  checkHealth();
  await loadSettings();
  setInterval(() => { if (!running) checkHealth(); }, 10000);

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    els.errTitle.textContent = 'Webcam not supported';
    els.errText.textContent =
      'This browser does not support camera access. Try a recent version of ' +
      'Chrome, Edge, Firefox or Safari over http://localhost.';
    els.retryBtn.hidden = true;
    showState('error');
    return;
  }

  if (els.video.srcObject && els.video.srcObject.active) {
    stream = els.video.srcObject;
    await waitForMetadata();
    if (!landmarker) await initLandmarker();
    goLive();
    return;
  }

  try {
    if (navigator.permissions && navigator.permissions.query) {
      const st = await navigator.permissions.query({ name: 'camera' });
      if (st.state === 'granted') startCamera();
    }
  } catch { /* permissions API is not everywhere; stay idle */ }
}

init();

// Exposed for the browser based end-to-end check.
window.__translator = {
  get running() { return running; },
  get landmarkerReady() { return !!landmarker; },
  postLandmarks,
  sessionId,
};
