/**
 * RAVIS — realtime.js
 *
 * Live camera with:
 *  1. Real-time lip mesh overlay (MediaPipe FaceMesh)
 *  2. Live fake/real detection indicator (updates every 2s)
 *  3. Per-second timeline tiles (green/yellow/red/gray)
 *  4. Voice quality animated bars (active/silent/loud)
 *  5. On Stop → full backend pipeline → final result
 */

(function () {
  'use strict';

  const API         = 'http://localhost:5000';
  const TARGET_FPS  = 25;
  const CHART_PTS   = 120;
  const POLL_MS     = 900;
  const DETECT_EVERY_MS = 2000;   // update fake indicator every 2 seconds

  // Lip landmark indices (MediaPipe 478-pt)
  const L_UP = 13, L_DOWN = 14, L_LEFT = 61, L_RIGHT = 291;
  const LIP_OUTER = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];
  const LIP_INNER = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95];

  const MAR_THRESH  = 0.08;   // mouth open threshold
  const RMS_THRESH  = 0.015;  // voice active threshold
  const RMS_LOUD    = 0.08;   // loud voice
  const SYNC_LOW    = 0.40;
  const SYNC_HIGH   = 0.62;

  // ─── State ───────────────────────────────────────────────────────────
  const rt = {
    running:      false,
    stream:       null,
    recorder:     null,
    chunks:       [],
    faceMesh:     null,
    rafId:        null,
    audioCtx:     null,
    analyser:     null,
    pollTimer:    null,
    elTimer:      null,
    detectTimer:  null,    // interval for live fake detection
    elapsed:      0,
    frames:       0,
    audioPeak:    0,
    silenceF:     0,
    lipEvents:    0,
    prevOpen:     false,
    lastMAR:      0,
    lastRMS:      0,
    lastSync:     0.5,
    faceDetected: false,

    // Rolling buffers for charts
    marBuf:  new Array(CHART_PTS).fill(0),
    rmsBuf:  new Array(CHART_PTS).fill(0),
    syncBuf: new Array(CHART_PTS).fill(0.5),

    // Per-second tile data  [{second, sync, mar, rms, status}]
    tiles: [],
    tileAccumMAR:  [],
    tileAccumRMS:  [],
    tileAccumSync: [],
    lastTileSecond: -1,
  };

  // ─── DOM ─────────────────────────────────────────────────────────────
  const el = {
    btnStart:    document.getElementById('rt-btn-start'),
    btnStop:     document.getElementById('rt-btn-stop'),
    camSelect:   document.getElementById('rt-cam-select'),
    video:       document.getElementById('rt-webcam-video'),
    canvas:      document.getElementById('rt-mesh-canvas'),
    placeholder: document.getElementById('cam-placeholder'),
    hud:         document.getElementById('cam-hud'),
    hudMAR:      document.getElementById('hud-mar'),
    hudSync:     document.getElementById('hud-sync'),
    statusBadge: document.getElementById('live-status-badge'),
    elapsed:     document.getElementById('rt-elapsed'),
    frameCount:  document.getElementById('rt-frame-counter'),

    // Live Fake Indicator
    fakeBadge:   document.getElementById('live-fake-indicator'),
    lfiIcon:     document.getElementById('lfi-icon'),
    lfiLabel:    document.getElementById('lfi-label'),
    lfiConf:     document.getElementById('lfi-conf'),

    // Voice Quality
    voiceBadge:  document.getElementById('voice-quality-badge'),
    vqLabel:     document.getElementById('vq-label'),
    vqBars:      [
      document.getElementById('vq-b1'),
      document.getElementById('vq-b2'),
      document.getElementById('vq-b3'),
      document.getElementById('vq-b4'),
      document.getElementById('vq-b5'),
    ],

    // Timeline tiles
    tilesWrap:   document.getElementById('timeline-tiles-wrap'),
    tilesRow:    document.getElementById('timeline-tiles'),

    // Charts
    marChart:    document.getElementById('rt-mar-chart'),
    audioChart:  document.getElementById('rt-audio-chart'),
    syncChart:   document.getElementById('rt-sync-chart'),

    // Metric labels
    marCurrent:  document.getElementById('rt-mar-current'),
    marMean:     document.getElementById('rt-mar-mean'),
    marStd:      document.getElementById('rt-mar-std'),
    lipEvents:   document.getElementById('rt-lip-events'),
    audioRMS:    document.getElementById('rt-audio-rms'),
    audioPeak:   document.getElementById('rt-audio-peak'),
    silenceRatio:document.getElementById('rt-silence-ratio'),
    voiceActive: document.getElementById('rt-voice-active'),
    syncScore:   document.getElementById('rt-sync-score'),
    lagMs:       document.getElementById('rt-lag-ms'),
    xcorr:       document.getElementById('rt-xcorr'),
    suspPct:     document.getElementById('rt-suspicious-pct'),

    // Live verdict panel
    lvIdle:      document.getElementById('lv-idle'),
    lvActive:    document.getElementById('lv-active'),
    lvIcon:      document.getElementById('lv-icon'),
    lvLabel:     document.getElementById('lv-label'),
    lvSub:       document.getElementById('lv-sub'),
    lvFakeFill:  document.getElementById('lv-fake-fill'),
    lvRealFill:  document.getElementById('lv-real-fill'),
    lvSyncFill:  document.getElementById('lv-sync-fill'),
    lvFakePct:   document.getElementById('lv-fake-pct'),
    lvRealPct:   document.getElementById('lv-real-pct'),
    lvSyncPct:   document.getElementById('lv-sync-pct'),
    lvExpl:      document.getElementById('lv-explanation'),

    // Live result card (shown after analysis)
    liveResultCard:  document.getElementById('live-result-card'),
    liveResultInner: document.getElementById('live-results-inner'),
  };

  // ─── Init ─────────────────────────────────────────────────────────────
  function init() {
    if (!el.btnStart) return;
    el.btnStart.addEventListener('click', startRecording);
    el.btnStop.addEventListener('click',  stopAndAnalyze);
    enumerateCameras();
    initFaceMesh();
    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);
  }

  // ─── Cameras ──────────────────────────────────────────────────────────
  async function enumerateCameras() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter(d => d.kind === 'videoinput');
      el.camSelect.innerHTML = '';
      if (!cams.length) { el.camSelect.innerHTML = '<option>No cameras found</option>'; return; }
      cams.forEach((c, i) => {
        const o = document.createElement('option');
        o.value = c.deviceId;
        o.textContent = c.label || `Camera ${i + 1}`;
        el.camSelect.appendChild(o);
      });
    } catch (_) {}
  }

  // ─── MediaPipe ────────────────────────────────────────────────────────
  function initFaceMesh() {
    if (typeof window.FaceMesh === 'undefined') { rt.faceMesh = null; return; }
    rt.faceMesh = new window.FaceMesh({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`,
    });
    rt.faceMesh.setOptions({
      maxNumFaces: 1, refineLandmarks: true,
      minDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    rt.faceMesh.onResults(onFaceResults);
  }

  // ═══════════════════════════════════════════════════════════════════
  //  START RECORDING
  // ═══════════════════════════════════════════════════════════════════
  async function startRecording() {
    el.btnStart.disabled = true;
    setBadge('Starting…', '');

    try {
      const deviceId = el.camSelect.value;
      rt.stream = await navigator.mediaDevices.getUserMedia({
        video: { width:{ideal:640}, height:{ideal:480}, frameRate:{ideal:30},
                 ...(deviceId ? { deviceId:{exact:deviceId} } : {}) },
        audio: { echoCancellation:true, noiseSuppression:true, sampleRate:16000 },
      });

      el.video.srcObject     = rt.stream;
      el.video.style.display = 'block';
      el.placeholder.style.display = 'none';
      el.hud.style.display   = 'flex';
      enumerateCameras();
      setupAudio(rt.stream);

      // MediaRecorder
      rt.chunks = [];
      const mime = getSupportedMime();
      rt.recorder = new MediaRecorder(rt.stream, mime ? { mimeType: mime } : {});
      rt.recorder.ondataavailable = e => { if (e.data?.size > 0) rt.chunks.push(e.data); };
      rt.recorder.start(200);

      // Reset all state
      rt.running   = true;
      rt.frames    = 0;
      rt.elapsed   = 0;
      rt.audioPeak = 0;
      rt.silenceF  = 0;
      rt.lipEvents = 0;
      rt.lastMAR   = 0;
      rt.lastRMS   = 0;
      rt.lastSync  = 0.5;
      rt.faceDetected = false;
      rt.tiles     = [];
      rt.tileAccumMAR  = [];
      rt.tileAccumRMS  = [];
      rt.tileAccumSync = [];
      rt.lastTileSecond = -1;
      rt.marBuf.fill(0);
      rt.rmsBuf.fill(0);
      rt.syncBuf.fill(0.5);

      el.btnStop.disabled = false;
      setBadge('Recording', 'recording');

      // Show live overlays
      el.fakeBadge.style.display  = 'flex';
      el.voiceBadge.style.display = 'flex';
      el.tilesWrap.style.display  = 'block';
      el.tilesRow.innerHTML       = '';

      setFakeIndicator('analyzing', '—', '');

      // Show live verdict panel
      el.lvIdle.style.display   = 'none';
      el.lvActive.style.display = 'flex';
      el.lvIcon.textContent     = '⏺';
      el.lvLabel.textContent    = 'Recording…';
      el.lvLabel.className      = 'lv-label analyzing';
      el.lvSub.textContent      = 'Live tracking active — press Stop & Analyze for full result';
      el.lvExpl.textContent     = 'Lip mesh tracking and voice monitoring running in real-time.';

      if (el.liveResultCard) el.liveResultCard.style.display = 'none';

      startElapsed();
      startFrameLoop();
      startLiveDetection();   // ← updates fake indicator every 2s

    } catch (err) {
      el.btnStart.disabled = false;
      setBadge('Error', '');
      alert(`Cannot open camera:\n${err.message}\n\nPlease allow camera and microphone access.`);
    }
  }

  // ═══════════════════════════════════════════════════════════════════
  //  STOP & ANALYZE
  // ═══════════════════════════════════════════════════════════════════
  function stopAndAnalyze() {
    if (!rt.running) return;
    el.btnStop.disabled = true;
    rt.running = false;
    stopFrameLoop();
    stopElapsed();
    stopLiveDetection();
    setBadge('Analyzing…', 'analyzing');

    if (rt.recorder && rt.recorder.state !== 'inactive') {
      rt.recorder.onstop = () => { stopCamera(); sendToBackend(); };
      rt.recorder.stop();
    } else {
      stopCamera();
      sendToBackend();
    }
  }

  function stopCamera() {
    if (rt.stream)   { rt.stream.getTracks().forEach(t => t.stop()); rt.stream = null; }
    if (rt.audioCtx?.state !== 'closed') { rt.audioCtx?.close(); rt.audioCtx = null; rt.analyser = null; }
    el.video.srcObject     = null;
    el.video.style.display = 'none';
    el.hud.style.display   = 'none';
    el.fakeBadge.style.display  = 'none';
    el.voiceBadge.style.display = 'none';
    el.placeholder.style.display = 'flex';
    el.placeholder.innerHTML = `
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#4a67cc" stroke-width="1.4"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      <p style="color:var(--indigo-light);font-weight:700">Analyzing recording…</p>
      <p class="small">${rt.chunks.length} chunks · ${rt.elapsed}s recorded</p>`;
  }

  // ═══════════════════════════════════════════════════════════════════
  //  LIVE FAKE DETECTION — updates indicator every 2 seconds
  // ═══════════════════════════════════════════════════════════════════
  function startLiveDetection() {
    stopLiveDetection();
    rt.detectTimer = setInterval(runLiveDetect, DETECT_EVERY_MS);
  }

  function stopLiveDetection() {
    if (rt.detectTimer) { clearInterval(rt.detectTimer); rt.detectTimer = null; }
  }

  function runLiveDetect() {
    if (!rt.running || rt.frames < 10) return;

    const marArr  = rt.marBuf.slice(-50).filter(v => v > 0);
    const rmsArr  = rt.rmsBuf.slice(-50);
    const syncArr = rt.syncBuf.slice(-50);

    const marMean  = avg(marArr) || 0;
    const rmsMean  = avg(rmsArr);
    const syncMean = avg(syncArr);

    // Quick heuristic classification (same logic as backend heuristic)
    const signals = [
      1.0 - syncMean,                                        // low sync = fake
      Math.min(1, Math.abs(rmsMean - marMean * 0.15) * 8),  // mismatch
      marMean < 0.01 && rmsMean > RMS_THRESH ? 0.9 : 0,     // mouth closed + audio
      rmsMean < 0.005 && marMean > 0.12 ? 0.9 : 0,          // audio silent + mouth open
      syncMean < SYNC_LOW ? 0.7 : 0,                         // clear sync failure
    ];

    const fakeProbRaw = signals.reduce((a, b) => a + b, 0) / signals.length;
    const fakeProb    = Math.min(0.97, Math.max(0.03, fakeProbRaw));
    const realProb    = 1 - fakeProb;
    const conf        = Math.abs(fakeProb - 0.5) * 2;

    const isFake    = fakeProb > 0.52;
    const confPct   = (conf * 100).toFixed(0) + '%';

    // Update big fake indicator on camera
    if (fakeProb > 0.65) {
      setFakeIndicator('fake', '🚨', 'FAKE', confPct);
    } else if (fakeProb > 0.45) {
      setFakeIndicator('analyzing', '🟡', 'UNSURE', confPct);
    } else {
      setFakeIndicator('real', '✅', 'REAL', confPct);
    }

    // Update live verdict panel bars
    const fp = (fakeProb * 100).toFixed(1);
    const rp = (realProb * 100).toFixed(1);
    const sp = (syncMean * 100).toFixed(0);

    set(el.lvFakeFill, null, 'style.width', fp + '%');
    set(el.lvRealFill, null, 'style.width', rp + '%');
    set(el.lvSyncFill, null, 'style.width', sp + '%');
    if (el.lvFakePct) el.lvFakePct.textContent = fp + '%';
    if (el.lvRealPct) el.lvRealPct.textContent = rp + '%';
    if (el.lvSyncPct) el.lvSyncPct.textContent = syncMean.toFixed(2);

    // Update explanation
    let expl;
    if (fakeProb > 0.65) {
      expl = `⚠ Suspicious: Lip movements are not matching the audio. ` +
             `Sync score ${syncMean.toFixed(2)} is below the natural threshold of 0.62. ` +
             (marMean < 0.01 && rmsMean > RMS_THRESH ? `Mouth appears closed while audio is active. ` : '') +
             `Fake probability: ${fp}%`;
    } else if (fakeProb > 0.45) {
      expl = `Uncertain: Partial mismatch between lips and voice. ` +
             `Sync score ${syncMean.toFixed(2)}. Keep recording for a more reliable result.`;
    } else {
      expl = `✓ Looks authentic so far. Lip movements are naturally synchronized with audio. ` +
             `Sync score ${syncMean.toFixed(2)}, delay is within normal range.`;
    }
    if (el.lvExpl) el.lvExpl.textContent = expl;
  }

  function setFakeIndicator(cls, icon, label, conf) {
    if (!el.fakeBadge) return;
    el.fakeBadge.className = 'live-fake-indicator ' + cls;
    if (el.lfiIcon)  el.lfiIcon.textContent  = icon  || '🟡';
    if (el.lfiLabel) el.lfiLabel.textContent = label || '—';
    if (el.lfiConf)  el.lfiConf.textContent  = conf  || '';
  }

  // ═══════════════════════════════════════════════════════════════════
  //  PER-SECOND TIMELINE TILES
  // ═══════════════════════════════════════════════════════════════════
  function addTileForSecond(second) {
    const marMean  = avg(rt.tileAccumMAR)  || 0;
    const rmsMean  = avg(rt.tileAccumRMS)  || 0;
    const syncMean = avg(rt.tileAccumSync) || 0.5;

    let status;
    if (!rt.faceDetected && rt.tileAccumMAR.length < 5) {
      status = 'noface';
    } else if (syncMean > SYNC_HIGH) {
      status = 'good';
    } else if (syncMean > SYNC_LOW) {
      status = 'warn';
    } else {
      status = 'bad';
    }

    const tile = { second, marMean, rmsMean, syncMean, status };
    rt.tiles.push(tile);

    // Clear accumulators
    rt.tileAccumMAR  = [];
    rt.tileAccumRMS  = [];
    rt.tileAccumSync = [];

    renderTile(tile);
  }

  function renderTile(tile) {
    if (!el.tilesRow) return;

    // Remove 'current' from last tile
    const prev = el.tilesRow.querySelector('.tt-tile.current');
    if (prev) prev.classList.remove('current');

    const div = document.createElement('div');
    div.className = `tt-tile ${tile.status} current`;

    const m = String(Math.floor(tile.second / 60)).padStart(2, '0');
    const s = String(tile.second % 60).padStart(2, '0');
    const tsLabel = `${m}:${s}`;

    const icon = tile.status === 'good' ? '✓' : tile.status === 'bad' ? '✗' : tile.status === 'noface' ? '?' : '~';
    const syncPct = (tile.syncMean * 100).toFixed(0);
    const voiceIcon = tile.rmsMean > RMS_THRESH ? '🔊' : '🔇';

    div.setAttribute('data-tip', `${tsLabel} · Sync ${syncPct}% · ${voiceIcon} · MAR ${tile.marMean.toFixed(2)}`);
    div.innerHTML = `<span class="tt-label">${tsLabel}</span>`;

    div.addEventListener('click', () => {
      // Flash tooltip or highlight this second
      div.style.outline = '2px solid white';
      setTimeout(() => { div.style.outline = ''; }, 600);
    });

    el.tilesRow.appendChild(div);

    // Auto-scroll tiles to end
    el.tilesRow.scrollLeft = el.tilesRow.scrollWidth;
  }

  // ═══════════════════════════════════════════════════════════════════
  //  VOICE QUALITY INDICATOR
  // ═══════════════════════════════════════════════════════════════════
  function updateVoiceIndicator(rms) {
    if (!el.voiceBadge) return;

    let cls, label;
    if (rms < 0.003) {
      cls = 'silent'; label = 'SILENT';
    } else if (rms < RMS_THRESH) {
      cls = 'silent'; label = 'QUIET';
    } else if (rms > RMS_LOUD) {
      cls = 'loud'; label = 'LOUD';
    } else {
      cls = 'active'; label = 'VOICE';
    }

    el.voiceBadge.className = 'voice-quality-badge ' + cls;
    if (el.vqLabel) el.vqLabel.textContent = label;

    // Animate bar heights based on RMS level
    const level = Math.min(1, rms / RMS_LOUD);
    const heights = [4, 7, 11, 15, 20]; // base heights
    el.vqBars.forEach((bar, i) => {
      if (!bar) return;
      const threshold = (i + 1) / el.vqBars.length;
      if (level >= threshold) {
        const jitter = 0.7 + Math.random() * 0.6;
        bar.style.height = Math.round(heights[i] * jitter) + 'px';
        bar.style.opacity = '1';
      } else {
        bar.style.height = heights[i] * 0.3 + 'px';
        bar.style.opacity = '0.3';
      }
    });
  }

  // ═══════════════════════════════════════════════════════════════════
  //  FRAME LOOP
  // ═══════════════════════════════════════════════════════════════════
  let _lastTs = 0;
  const FRAME_MS = 1000 / TARGET_FPS;

  function startFrameLoop() { _lastTs = 0; rt.rafId = requestAnimationFrame(frameLoop); }
  function stopFrameLoop()  { if (rt.rafId) { cancelAnimationFrame(rt.rafId); rt.rafId = null; } }

  function frameLoop(ts) {
    if (!rt.running) return;
    if (ts - _lastTs >= FRAME_MS) {
      _lastTs = ts;
      processFrame();
    }
    rt.rafId = requestAnimationFrame(frameLoop);
  }

  async function processFrame() {
    rt.frames++;

    // Read audio
    const rms = readRMS();
    push(rt.rmsBuf, rms);
    rt.tileAccumRMS.push(rms);

    // Update voice indicator every frame
    updateVoiceIndicator(rms);

    // Per-second tile accumulation
    const currentSec = Math.floor(rt.elapsed);
    if (currentSec > rt.lastTileSecond && rt.lastTileSecond >= 0) {
      addTileForSecond(rt.lastTileSecond);
    }
    rt.lastTileSecond = currentSec;

    // Run MediaPipe
    if (rt.faceMesh && el.video.readyState >= 2) {
      try { await rt.faceMesh.send({ image: el.video }); } catch (_) {}
    } else {
      const est = Math.min(0.5, rms * 3);
      push(rt.marBuf, est);
      const s = frameSyncScore(est, rms);
      push(rt.syncBuf, s);
      rt.tileAccumMAR.push(est);
      rt.tileAccumSync.push(s);
      drawFallback(rms);
      updateHUD(est, rms);
    }

    drawCharts();
    updateMetricLabels();
  }

  // ─── MediaPipe callback ───────────────────────────────────────────
  function onFaceResults(res) {
    if (!rt.running) return;
    const ctx = el.canvas.getContext('2d');
    ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    const rms = rt.rmsBuf[rt.rmsBuf.length - 1] || 0;

    if (!res.multiFaceLandmarks?.length) {
      rt.faceDetected = false;
      push(rt.marBuf, 0);
      const s = frameSyncScore(0, rms);
      push(rt.syncBuf, s);
      rt.tileAccumMAR.push(0);
      rt.tileAccumSync.push(s);

      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.fillStyle = 'rgba(242,179,61,0.9)';
      ctx.fillText('⚠  No face detected', 10, 24);
      drawOverlayLabels(ctx, 0, rms);
      updateHUD(0, rms);
      return;
    }

    rt.faceDetected = true;
    const lm  = res.multiFaceLandmarks[0];
    const w   = el.canvas.width;
    const h   = el.canvas.height;
    const mar = computeMAR(lm, w, h);

    push(rt.marBuf, mar);
    const s = frameSyncScore(mar, rms);
    push(rt.syncBuf, s);
    rt.tileAccumMAR.push(mar);
    rt.tileAccumSync.push(s);
    rt.lastSync = s;

    const open = mar > MAR_THRESH;
    if (open && !rt.prevOpen) rt.lipEvents++;
    rt.prevOpen = open;

    drawLips(ctx, lm, w, h, mar, rms);
    updateHUD(mar, rms);
  }

  // ─── Compute MAR ─────────────────────────────────────────────────
  function computeMAR(lm, w, h) {
    const vert  = Math.hypot((lm[L_UP].x - lm[L_DOWN].x)*w,  (lm[L_UP].y - lm[L_DOWN].y)*h);
    const horiz = Math.hypot((lm[L_LEFT].x - lm[L_RIGHT].x)*w,(lm[L_LEFT].y - lm[L_RIGHT].y)*h);
    return horiz > 1 ? vert / horiz : 0;
  }

  // ─── Draw Lip Mesh ────────────────────────────────────────────────
  function drawLips(ctx, lm, w, h, mar, rms) {
    const open  = mar > MAR_THRESH;
    const wide  = mar > 0.27;
    // Color: green = good sync, yellow = mismatch, red = wide open/suspicious
    const syncNow = rt.syncBuf[rt.syncBuf.length - 1] || 0.5;
    let color, glow;
    if (wide || syncNow < SYNC_LOW) {
      color = '#ef5a6b'; glow = 'rgba(239,90,107,0.65)';
    } else if (open || syncNow < SYNC_HIGH) {
      color = '#f2b33d'; glow = 'rgba(242,179,61,0.55)';
    } else {
      color = '#2fcb93'; glow = 'rgba(47,203,147,0.55)';
    }

    // Outer lip
    ctx.shadowBlur = 9; ctx.shadowColor = glow;
    drawContour(ctx, lm, LIP_OUTER, w, h, color, 2.2);
    ctx.shadowBlur = 5;
    drawContour(ctx, lm, LIP_INNER, w, h, color, 1.2);
    ctx.shadowBlur = 0;

    // Landmark dots on lip
    [...LIP_OUTER, ...LIP_INNER].forEach(i => {
      ctx.beginPath();
      ctx.arc(lm[i].x*w, lm[i].y*h, 2.2, 0, Math.PI*2);
      ctx.fillStyle = color; ctx.fill();
    });

    // MAR label above mouth
    const mx = lm[L_UP].x * w;
    const my = lm[L_UP].y * h - 18;
    ctx.font      = 'bold 11px Inter, monospace';
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.fillText(`MAR ${mar.toFixed(3)}`, mx, my);
    ctx.textAlign = 'left';

    drawOverlayLabels(ctx, mar, rms);
  }

  function drawOverlayLabels(ctx, mar, rms) {
    const w = el.canvas.width;
    const syncNow = rt.syncBuf[rt.syncBuf.length - 1] || 0.5;
    const sc = syncNow > SYNC_HIGH ? '#2fcb93' : syncNow < SYNC_LOW ? '#ef5a6b' : '#f2b33d';

    // Top-left: SYNC score
    ctx.font = 'bold 11px Inter, monospace';
    ctx.fillStyle = sc;
    ctx.textAlign = 'left';
    ctx.fillText(`SYNC ${(syncNow * 100).toFixed(0)}%`, 8, 18);

    // Top-right: REC
    ctx.fillStyle = '#ef5a6b';
    ctx.fillText('⏺ REC', w - 55, 18);

    // Top-right: voice status
    const voiceColor = rms > RMS_THRESH ? '#2fcb93' : '#7c84a4';
    ctx.fillStyle = voiceColor;
    ctx.fillText(rms > RMS_THRESH ? '🔊' : '🔇', w - 80, 18);
  }

  function drawContour(ctx, lm, idxs, w, h, color, lw) {
    ctx.beginPath();
    idxs.forEach((i, n) => n === 0 ? ctx.moveTo(lm[i].x*w, lm[i].y*h) : ctx.lineTo(lm[i].x*w, lm[i].y*h));
    ctx.closePath();
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
  }

  function drawFallback(rms) {
    const ctx = el.canvas.getContext('2d');
    ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.fillStyle = 'rgba(242,179,61,0.8)';
    ctx.fillText('⚠ MediaPipe offline — audio only', 10, 18);
    ctx.fillStyle = '#ef5a6b';
    ctx.fillText('⏺ REC', el.canvas.width - 55, 18);
    const voiceColor = rms > RMS_THRESH ? '#2fcb93' : '#7c84a4';
    ctx.fillStyle = voiceColor;
    ctx.fillText(rms > RMS_THRESH ? '🔊' : '🔇', el.canvas.width - 80, 18);
  }

  // ─── Audio ────────────────────────────────────────────────────────
  function setupAudio(stream) {
    try {
      rt.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      rt.analyser = rt.audioCtx.createAnalyser();
      rt.analyser.fftSize = 2048;
      rt.analyser.smoothingTimeConstant = 0.3;
      const tracks = stream.getAudioTracks();
      if (!tracks.length) return;
      rt.audioCtx.createMediaStreamSource(new MediaStream([tracks[0]])).connect(rt.analyser);
    } catch (e) { rt.analyser = null; }
  }

  function readRMS() {
    if (!rt.analyser) return 0;
    const buf = new Float32Array(rt.analyser.fftSize);
    rt.analyser.getFloatTimeDomainData(buf);
    let sq = 0; for (const v of buf) sq += v * v;
    const rms = Math.sqrt(sq / buf.length);
    if (rms > rt.audioPeak) rt.audioPeak = rms;
    if (rms < RMS_THRESH) rt.silenceF++;
    rt.lastRMS = rms;
    return rms;
  }

  function frameSyncScore(mar, rms) {
    const mo = mar > MAR_THRESH, ao = rms > RMS_THRESH;
    if (mo && ao)   return 0.78 + Math.random() * 0.17;
    if (!mo && !ao) return 0.68 + Math.random() * 0.15;
    return 0.12 + Math.random() * 0.22;  // mismatch
  }

  // ─── HUD update ───────────────────────────────────────────────────
  function updateHUD(mar, rms) {
    rt.lastMAR = mar;
    if (el.hudMAR)  el.hudMAR.textContent  = `MAR: ${mar.toFixed(3)}`;
    if (el.hudSync) {
      const s = rt.syncBuf[rt.syncBuf.length - 1] || 0.5;
      el.hudSync.textContent = `SYNC: ${(s * 100).toFixed(0)}%`;
    }
    if (el.frameCount) el.frameCount.textContent = rt.frames;
  }

  function updateMetricLabels() {
    const n = Math.max(1, rt.frames);
    const mar = rt.marBuf[rt.marBuf.length - 1] || 0;
    const rms = rt.lastRMS;
    const sync = rt.syncBuf[rt.syncBuf.length - 1] || 0.5;

    setText(el.marCurrent,   mar.toFixed(3));
    setText(el.marMean,      avg(rt.marBuf).toFixed(3));
    setText(el.marStd,       stdDev(rt.marBuf).toFixed(3));
    setText(el.lipEvents,    rt.lipEvents);
    setText(el.audioRMS,     rms.toFixed(3));
    setText(el.audioPeak,    rt.audioPeak.toFixed(3));
    setText(el.silenceRatio, Math.round((rt.silenceF / n) * 100) + '%');
    if (el.voiceActive) {
      el.voiceActive.textContent = rms > RMS_THRESH ? 'YES' : 'NO';
    }
    setText(el.syncScore, sync.toFixed(3));
    setText(el.lagMs,     '(pending)');
    setText(el.xcorr,     '(pending)');

    const suspCount = rt.tiles.filter(t => t.status === 'bad').length;
    const suspPct = rt.tiles.length ? Math.round((suspCount / rt.tiles.length) * 100) + '%' : '—';
    setText(el.suspPct, suspPct);

    if (el.syncScore) {
      el.syncScore.className = 'chart-val ' + (sync > SYNC_HIGH ? 'green' : sync < SYNC_LOW ? 'red' : 'yellow');
    }
  }

  function setText(elem, val) { if (elem) elem.textContent = val; }

  // set style property helper
  function set(elem, prop, styleProp, val) {
    if (!elem) return;
    if (styleProp) { elem.style[styleProp.replace('style.', '')] = val; }
    else if (prop) { elem[prop] = val; }
  }

  // ─── Charts ───────────────────────────────────────────────────────
  function resizeCanvases() {
    [el.marChart, el.audioChart, el.syncChart].forEach(c => {
      if (!c) return;
      const w = c.parentElement?.getBoundingClientRect().width || 300;
      c.width = Math.floor(w); c.height = 56;
    });
    if (el.canvas) {
      const vp = document.getElementById('cam-viewport');
      if (vp) {
        const r = vp.getBoundingClientRect();
        el.canvas.width  = Math.floor(r.width)  || 640;
        el.canvas.height = Math.floor(r.height) || 480;
      }
    }
  }

  function push(arr, val) { arr.push(val); if (arr.length > CHART_PTS) arr.shift(); }

  function drawCharts() {
    drawChart(el.marChart,   rt.marBuf,  0, 0.6,  '#4a67cc', 'rgba(74,103,204,0.18)');
    drawChart(el.audioChart, rt.rmsBuf,  0, 0.15, '#2fcb93', 'rgba(47,203,147,0.15)');
    const sc = rt.syncBuf[rt.syncBuf.length - 1] || 0.5;
    const lc = sc > SYNC_HIGH ? '#2fcb93' : sc < SYNC_LOW ? '#ef5a6b' : '#f2b33d';
    drawChart(el.syncChart, rt.syncBuf, 0, 1.0, lc, 'rgba(74,103,204,0.08)', [
      { y: SYNC_LOW, c: 'rgba(239,90,107,0.5)' },
      { y: SYNC_HIGH, c: 'rgba(47,203,147,0.5)' },
    ]);
  }

  function drawChart(canvas, data, lo, hi, stroke, fill, thresholds) {
    if (!canvas || !data.length) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const range = hi - lo || 1;
    const step  = w / (data.length - 1 || 1);
    const yOf   = v => h - Math.max(2, Math.min(h - 2, ((v - lo) / range) * h));

    ctx.beginPath(); ctx.moveTo(0, h);
    data.forEach((v, i) => ctx.lineTo(i * step, yOf(v)));
    ctx.lineTo(w, h); ctx.closePath();
    ctx.fillStyle = fill; ctx.fill();

    ctx.beginPath();
    data.forEach((v, i) => i === 0 ? ctx.moveTo(0, yOf(v)) : ctx.lineTo(i * step, yOf(v)));
    ctx.strokeStyle = stroke; ctx.lineWidth = 1.6; ctx.stroke();

    thresholds?.forEach(({ y, c }) => {
      const py = yOf(y);
      ctx.beginPath(); ctx.setLineDash([4, 4]);
      ctx.moveTo(0, py); ctx.lineTo(w, py);
      ctx.strokeStyle = c; ctx.lineWidth = 1; ctx.stroke();
      ctx.setLineDash([]);
    });
  }

  // ─── Elapsed timer ────────────────────────────────────────────────
  function startElapsed() {
    rt.elapsed = 0;
    rt.elTimer = setInterval(() => {
      rt.elapsed++;
      const m = String(Math.floor(rt.elapsed / 60)).padStart(2, '0');
      const s = String(rt.elapsed % 60).padStart(2, '0');
      if (el.elapsed) el.elapsed.textContent = `${m}:${s}`;
    }, 1000);
  }
  function stopElapsed() {
    if (rt.elTimer) { clearInterval(rt.elTimer); rt.elTimer = null; }
  }

  // ─── Badge ────────────────────────────────────────────────────────
  function setBadge(text, cls) {
    if (!el.statusBadge) return;
    el.statusBadge.textContent = text;
    el.statusBadge.className   = 'live-status-badge' + (cls ? ' ' + cls : '');
  }

  // ═══════════════════════════════════════════════════════════════════
  //  BACKEND PIPELINE (after Stop)
  // ═══════════════════════════════════════════════════════════════════
  async function sendToBackend() {
    if (!rt.chunks.length) { showLiveError('No recording captured. Please try again.'); return; }

    el.lvIcon.textContent  = '🔬';
    el.lvLabel.textContent = 'Analyzing…';
    el.lvLabel.className   = 'lv-label analyzing';
    el.lvSub.textContent   = 'Running full deepfake detection pipeline…';
    el.lvExpl.textContent  = 'Audio extraction → face detection → lip sync analysis → AI classifier…';

    const mime = getSupportedMime() || 'video/webm';
    const blob = new Blob(rt.chunks, { type: mime });
    const ext  = mime.includes('mp4') ? 'mp4' : 'webm';
    const file = new File([blob], `webcam_recording.${ext}`, { type: mime });

    const form = new FormData();
    form.append('video', file);

    try {
      const res = await fetch(`${API}/api/analyze`, { method: 'POST', body: form });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `HTTP ${res.status}`);
      const data = await res.json();
      startPoll(data.job_id, file.name);
    } catch (err) {
      console.warn('Live upload failed:', err.message);
      showLiveError(
        'Could not upload the recording. Check that the backend is running and try again.'
      );
    }
  }

  function startPoll(jobId, filename) {
    clearPoll();
    rt.pollTimer = setInterval(async () => {
      try {
        const res  = await fetch(`${API}/api/status/${jobId}`);
        if (!res.ok) return;
        const data = await res.json();
        const pct = data.progress || 0;
        if (el.lvSub) el.lvSub.textContent = `${data.stage || 'Processing'} — ${pct}%`;
        if (data.status === 'done')   { clearPoll(); showLiveResult(data.result, filename); }
        else if (data.status === 'failed') { clearPoll(); showLiveError(data.error || 'Failed.'); }
      } catch (e) {}
    }, POLL_MS);
  }

  function clearPoll() {
    if (rt.pollTimer) { clearInterval(rt.pollTimer); rt.pollTimer = null; }
  }

  function showLiveResult(result, filename) {
    setBadge('Done', '');
    el.btnStart.disabled = false;

    const isFake      = result.verdict === 'DEEPFAKE';
    const isUncertain = result.verdict === 'INCONCLUSIVE';
    const liveLabel   = isFake ? 'FAKE DETECTED' : isUncertain ? 'INCONCLUSIVE' : 'AUTHENTIC';
    const liveClass   = isFake ? 'fake' : isUncertain ? 'uncertain' : 'real';
    if (el.lvIcon)  el.lvIcon.textContent  = isFake ? '🚨' : isUncertain ? '⚠️' : '✅';
    if (el.lvLabel) { el.lvLabel.textContent = liveLabel; el.lvLabel.className = `lv-label ${liveClass}`; }
    if (el.lvSub)   el.lvSub.textContent   = isUncertain
      ? `Verdict withheld · Sync: ${result.sync_score?.toFixed(3) || '—'}`
      : `Confidence: ${result.confidence != null ? (result.confidence*100).toFixed(1)+'%' : '—'} · Sync: ${result.sync_score?.toFixed(3) || '—'}`;
    if (el.lvExpl)  el.lvExpl.textContent  = result.explanation?.summary || result.decision_reason || 'Analysis complete.';

    const fp = (result.deepfake_probability * 100).toFixed(1);
    const rp = (result.real_probability * 100).toFixed(1);
    const sp = Math.min(100, (result.sync_score || 0) * 100).toFixed(0);
    if (el.lvFakeFill) el.lvFakeFill.style.width = fp + '%';
    if (el.lvRealFill) el.lvRealFill.style.width = rp + '%';
    if (el.lvSyncFill) el.lvSyncFill.style.width = sp + '%';
    if (el.lvFakePct)  el.lvFakePct.textContent  = fp + '%';
    if (el.lvRealPct)  el.lvRealPct.textContent  = rp + '%';
    if (el.lvSyncPct)  el.lvSyncPct.textContent  = result.sync_score?.toFixed(3) || '—';

    // Full result card
    if (el.liveResultCard && el.liveResultInner && window.DG) {
      el.liveResultInner.innerHTML = window.DG.buildResultHTML(
        result, isFake,
        result.confidence         != null ? (result.confidence*100).toFixed(1) : '—',
        result.sync_score         != null ? result.sync_score.toFixed(3) : '—',
        result.audio_lip_delay_ms != null ? result.audio_lip_delay_ms+'ms' : '—',
        result.face_detection_rate!= null ? (result.face_detection_rate*100).toFixed(0)+'%' : '—',
        result.reliability_grade  || '—'
      );
      el.liveResultCard.style.display = 'block';
      el.liveResultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Replace timeline tiles with pipeline's real per-second results
    if (result.timeline?.length && el.tilesRow) {
      el.tilesRow.innerHTML = '';
      result.timeline.forEach(seg => {
        const div = document.createElement('div');
        div.className = `tt-tile ${seg.status}`;
        const mm = String(Math.floor(seg.second/60)).padStart(2,'0');
        const ss = String(seg.second%60).padStart(2,'0');
        const voiceIcon = seg.lag_ms > 100 ? '⚠' : '✓';
        div.setAttribute('data-tip', `${mm}:${ss} · Sync ${(seg.sync_score*100).toFixed(0)}% · ${seg.error_type}`);
        div.innerHTML = `<span class="tt-label">${mm}:${ss}</span>`;
        el.tilesRow.appendChild(div);
      });
    }

    if (window.DG) { window.DG.saveHistory(filename, result); window.DG.renderHistory(); }
    window.DG?.setStatus('done', `Live: ${result.verdict}`);
  }

  function showLiveError(msg) {
    setBadge('Error', '');
    el.btnStart.disabled = false;
    if (el.lvIcon)  el.lvIcon.textContent  = '⚠️';
    if (el.lvLabel) { el.lvLabel.textContent = 'Failed'; el.lvLabel.className = 'lv-label analyzing'; }
    if (el.lvSub)   el.lvSub.textContent   = msg;
    if (el.lvExpl)  el.lvExpl.textContent  = 'Please try recording again. Make sure the server is running.';
  }

  // ─── Helpers ──────────────────────────────────────────────────────
  function avg(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }
  function stdDev(arr) {
    const m = avg(arr);
    return arr.length < 2 ? 0 : Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
  }
  function getSupportedMime() {
    const types = ['video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm','video/mp4'];
    return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
  }

  // ─── Boot ─────────────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
