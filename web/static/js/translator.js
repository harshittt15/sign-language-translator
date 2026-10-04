/* Real-Time Sign Language Translator, browser client.
 *
 * Captures webcam frames, posts them to the Python backend, and renders the
 * SMOOTHED prediction. The raw per frame prediction is shown only as a dim
 * secondary line so a noisy frame never looks like a committed letter.
 */
(() => {
  'use strict';

  const TARGET_FPS = 10;
  const MIN_INTERVAL = 1000 / TARGET_FPS;
  const JPEG_QUALITY = 0.72;
  const METADATA_TIMEOUT = 4000;

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
  let running = false;
  let inFlight = false;
  let audioOn = false;
  let transcript = '';
  let lastSent = 0;
  const captureCanvas = document.createElement('canvas');
  const rtt = [];

  /* ---------------- view state ----------------
   * One function owns which overlay is visible, so the camera can never be
   * live while a "camera is off" panel still covers it.
   */

  function showState(name) {
    els.stateIdle.hidden  = name !== 'idle';
    els.stateBusy.hidden  = name !== 'busy';
    els.stateError.hidden = name !== 'error';
    els.cropGuide.hidden  = name !== 'live';
  }

  function setDot(dot, cls) {
    dot.classList.remove('is-on', 'is-off', 'is-wait');
    if (cls) dot.classList.add(cls);
  }

  function setCamera(state, label) {
    setDot(els.dotCam, state);
    els.txtCam.textContent = label;
  }

  function setHand(detected) {
    setDot(els.dotHand, detected ? 'is-on' : null);
    els.txtHand.textContent = detected ? 'Hand detected' : 'No hand';
  }

  function setApi(state, label) {
    setDot(els.dotApi, state);
    els.txtApi.textContent = label;
  }

  /* ---------------- backend ---------------- */

  async function checkHealth() {
    try {
      const r = await fetch('/health');
      const b = await r.json();
      if (b.status === 'ok' && b.model_loaded) {
        setApi('is-on', 'Backend ready');
      } else {
        setApi('is-off', 'Model not loaded');
      }
    } catch {
      setApi('is-off', 'Backend unreachable');
    }
  }

  /* ---------------- camera ---------------- */

  async function waitForMetadata() {
    if (els.video.readyState >= 1 && els.video.videoWidth > 0) return;
    await new Promise((resolve) => {
      let settled = false;
      const done = () => {
        if (settled) return;
        settled = true;
        els.video.removeEventListener('loadedmetadata', done);
        clearTimeout(timer);
        resolve();
      };
      const timer = setTimeout(done, METADATA_TIMEOUT);
      els.video.addEventListener('loadedmetadata', done);
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
    } catch (err) {
      failCamera(err);
      return;
    }

    try {
      stream = s;
      els.video.srcObject = s;

      // The stream is live only once the element reports real dimensions.
      // Flipping the UI before this leaves a blank frame under the overlay.
      await waitForMetadata();
      try { await els.video.play(); } catch { /* autoplay may already be running */ }

      goLive();
    } catch (err) {
      failCamera(err);
    }
  }

  function goLive() {
    running = true;
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

    const name = err && err.name;
    let title = 'Camera unavailable';
    let text = 'Something stopped the camera from starting. Please try again.';

    if (name === 'NotAllowedError' || name === 'SecurityError') {
      title = 'Camera permission denied';
      text = 'Your browser blocked camera access. Allow it from the permissions ' +
             'menu in the address bar, then try again.';
    } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
      title = 'No camera found';
      text = 'No webcam was detected. Connect one and try again.';
    } else if (name === 'NotReadableError' || name === 'AbortError') {
      title = 'Camera already in use';
      text = 'Another application is using the webcam. Close it, then try again.';
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

  /* ---------------- capture loop ---------------- */

  function loop() {
    if (!running) return;
    const now = performance.now();
    if (!inFlight && now - lastSent >= MIN_INTERVAL && els.video.videoWidth > 0) {
      lastSent = now;
      sendFrame();
    }
    requestAnimationFrame(loop);
  }

  async function sendFrame() {
    inFlight = true;
    const t0 = performance.now();
    try {
      const vw = els.video.videoWidth, vh = els.video.videoHeight;
      captureCanvas.width = vw;
      captureCanvas.height = vh;
      // Draw UNMIRRORED. The preview is flipped in CSS for a natural self
      // view, but the model was trained on unmirrored hands and sending a
      // mirrored frame destroys accuracy.
      captureCanvas.getContext('2d').drawImage(els.video, 0, 0, vw, vh);

      const blob = await new Promise((res) =>
        captureCanvas.toBlob(res, 'image/jpeg', JPEG_QUALITY));
      if (!blob) { inFlight = false; return; }

      const fd = new FormData();
      fd.append('frame', blob, 'frame.jpg');
      fd.append('session_id', sessionId);
      fd.append('want_landmarks', 'true');

      const resp = await fetch('/predict', { method: 'POST', body: fd });
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const body = await resp.json();

      setApi('is-on', 'Backend ready');
      handleResult(body);
      trackRtt(performance.now() - t0);
    } catch {
      setApi('is-off', 'Connection lost');
    } finally {
      inFlight = false;
    }
  }

  function trackRtt(ms) {
    rtt.push(ms);
    if (rtt.length > 20) rtt.shift();
    const avg = rtt.reduce((a, b) => a + b, 0) / rtt.length;
    els.txtFps.textContent = (1000 / Math.max(avg, MIN_INTERVAL)).toFixed(1) + ' fps';
  }

  /* ---------------- rendering ---------------- */

  function handleResult(body) {
    setHand(!!body.hand_detected);
    showPrediction(body.sign, body.raw_sign, body.confidence || 0);
    drawLandmarks(body.landmarks);

    // `emitted` is true only on the frame the smoother commits a letter.
    if (body.emitted && body.sign) {
      transcript += body.sign;
      els.transcript.textContent = transcript;
      els.charCount.textContent =
        transcript.length + (transcript.length === 1 ? ' character' : ' characters');
      speak(body.sign);
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

  function drawLandmarks(points) {
    const c = els.overlay;
    const rect = c.getBoundingClientRect();
    if (c.width !== Math.round(rect.width) || c.height !== Math.round(rect.height)) {
      c.width = Math.round(rect.width);
      c.height = Math.round(rect.height);
    }
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    if (!points || !points.length) return;

    const vw = els.video.videoWidth, vh = els.video.videoHeight;
    if (!vw || !vh) return;

    // The backend analysed a centred square of the video and the landmarks
    // are [0,1] inside that square. Map square to video pixels, then apply
    // the CSS object-fit cover transform and the mirrored preview.
    const side = Math.min(vw, vh);
    const cropX = (vw - side) / 2, cropY = (vh - side) / 2;
    const scale = Math.max(c.width / vw, c.height / vh);
    const ox = (c.width - vw * scale) / 2;
    const oy = (c.height - vh * scale) / 2;

    const pts = points.map(([lx, ly]) => [
      c.width - (ox + (cropX + lx * side) * scale),
      oy + (cropY + ly * side) * scale,
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

  /* ---------------- audio ---------------- */

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

  /* ---------------- clear ---------------- */

  async function clearText() {
    transcript = '';
    els.transcript.textContent = '';
    els.charCount.textContent = '';
    try {
      const fd = new FormData();
      fd.append('session_id', sessionId);
      await fetch('/reset', { method: 'POST', body: fd });
    } catch { /* clearing locally is still correct if the reset call fails */ }
  }

  /* ---------------- startup ---------------- */

  els.startBtn.addEventListener('click', startCamera);
  els.retryBtn.addEventListener('click', startCamera);
  els.stopBtn.addEventListener('click', stopCamera);
  els.audioBtn.addEventListener('click', toggleAudio);
  els.clearBtn.addEventListener('click', clearText);
  window.addEventListener('beforeunload', () => { if (stream) stopCamera(); });

  // If the stream drops on its own, for example the device is unplugged,
  // return the UI to the off state instead of showing a frozen frame.
  els.video.addEventListener('emptied', () => { if (running) stopCamera(); });

  async function init() {
    showState('idle');
    setCamera(null, 'Camera off');
    checkHealth();
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

    // An element that already has a live stream, for example after a soft
    // reload, should show the live UI rather than the off state.
    if (els.video.srcObject && els.video.srcObject.active) {
      stream = els.video.srcObject;
      await waitForMetadata();
      goLive();
      return;
    }

    // If camera permission was granted earlier, resume without a second click.
    try {
      if (navigator.permissions && navigator.permissions.query) {
        const status = await navigator.permissions.query({ name: 'camera' });
        if (status.state === 'granted') startCamera();
      }
    } catch { /* permissions API is not available everywhere; stay idle */ }
  }

  init();
})();
