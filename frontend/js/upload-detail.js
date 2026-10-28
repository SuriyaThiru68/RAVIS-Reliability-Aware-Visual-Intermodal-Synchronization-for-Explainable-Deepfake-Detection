/* ═══════════════════════════════════════════════════════════════
   Upload page detail
   ─ Video player for the selected file
   ─ 3D Face Landmarker (MediaPipe Tasks Vision, 478 pts + blendshapes)
     with graceful fallback to the legacy FaceMesh solution
   ─ Kinetic tracking: motion vectors, energy, micro-jitter
   ─ Head-pose estimation (yaw / pitch / roll) + orientation gizmo
   ─ A/V synchronisation chart + forensic comparison table
   ─ Client-side SHA-256 for chain of custody
   ═══════════════════════════════════════════════════════════════ */
(() => {
  'use strict';

  const $ = id => document.getElementById(id);

  const stage      = $('mesh-stage');
  const canvas     = $('mesh-canvas');
  const ctx        = canvas ? canvas.getContext('2d') : null;
  const video      = $('up-video');
  const meshEmpty  = $('mesh-empty');
  const meshHud    = $('mesh-hud');
  const meshState  = $('mesh-status');
  const marVal     = $('mesh-mar');
  const fpsVal     = $('mesh-fps');
  const faceChip   = $('mesh-face-chip');
  const engineChip = $('mesh-engine-chip');

  const poseEls = {
    yaw:   { bar: $('pose-yaw'),   val: $('pose-yaw-val') },
    pitch: { bar: $('pose-pitch'), val: $('pose-pitch-val') },
    roll:  { bar: $('pose-roll'),  val: $('pose-roll-val') },
  };
  const jawEl     = $('mesh-jaw');
  const kineticEl = $('mesh-kinetic');
  const jitterEl  = $('mesh-jitter');
  const blinkEl   = $('mesh-blink');
  const pointsEl  = $('mesh-conf');
  const vectorsEl = $('mesh-vectors');

  // MediaPipe landmark indices
  const L_UP = 13, L_DOWN = 14, L_LEFT = 61, L_RIGHT = 291;
  const LIP_OUTER = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];
  const KINETIC_IDX = [];
  for (let i = 0; i < 478; i += 7) KINETIC_IDX.push(i);

  // ── State ──────────────────────────────────────────────────────────
  let objectUrl      = null;
  let engine         = null;     // 'tasks' | 'legacy'
  let engineState    = 'idle';   // idle | loading | ready | failed
  let visionRef      = null;
  let landmarker     = null;
  let legacyMesh     = null;

  let running        = false;
  let lastSend       = 0;
  let frames         = 0;
  let fpsStamp       = 0;
  let prevLm         = null;
  let prevPrevMean   = 0;
  let kineticEMA     = 0;
  let jitterEMA      = 0;
  let activeVectors  = 0;
  let lastResult     = null;
  let fileHash       = null;

  // ── Helpers ────────────────────────────────────────────────────────
  function setStatus(text, cls) {
    if (!meshState) return;
    meshState.textContent = text;
    meshState.className = 'live-status-badge' + (cls ? ' ' + cls : '');
  }

  function sizeCanvas() {
    if (!stage || !canvas) return;
    const r = stage.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(r.width * dpr));
    const h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function videoRect() {
    const vw = (video && video.videoWidth) || 16;
    const vh = (video && video.videoHeight) || 9;
    const cw = canvas.width, ch = canvas.height;
    const scale = Math.min(cw / vw, ch / vh);
    const w = vw * scale, h = vh * scale;
    return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
  }

  function paintFrame() {
    if (!ctx || !video || !video.videoWidth) return;
    const d = videoRect();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    try { ctx.drawImage(video, d.x, d.y, d.w, d.h); } catch (_) {}
  }

  function setPose(key, deg, range) {
    const el = poseEls[key];
    if (!el) return;
    range = range || 60;
    if (el.val) el.val.textContent = Math.round(deg) + '°';
    if (!el.bar) return;
    const clamped = Math.max(-range, Math.min(range, deg));
    const pct = (Math.abs(clamped) / range) * 50;
    if (clamped >= 0) {
      el.bar.style.left = '50%';
      el.bar.style.width = pct + '%';
    } else {
      el.bar.style.left = (50 - pct) + '%';
      el.bar.style.width = pct + '%';
    }
  }

  // ── Pose estimation ────────────────────────────────────────────────
  function poseFromMatrix(data) {
    if (!data || data.length < 16) return null;
    // Column-major 4×4 → rotation elements
    const r02 = data[8],  r12 = data[9],  r22 = data[10];
    const r10 = data[1],  r11 = data[5];
    const yaw   = Math.atan2(r02, r22) * 180 / Math.PI;
    const pitch = Math.asin(Math.max(-1, Math.min(1, -r12))) * 180 / Math.PI;
    const roll  = Math.atan2(r10, r11) * 180 / Math.PI;
    if (!isFinite(yaw) || !isFinite(pitch) || !isFinite(roll)) return null;
    return { yaw, pitch, roll };
  }

  function poseFromLandmarks(lm) {
    const nose = lm[1], le = lm[33], re = lm[263], chin = lm[152], top = lm[10];
    if (!nose || !le || !re || !chin || !top) return { yaw: 0, pitch: 0, roll: 0 };
    const roll = Math.atan2(re.y - le.y, re.x - le.x) * 180 / Math.PI;
    const eyeSpan = Math.abs(re.x - le.x) || 0.001;
    const yaw = ((nose.x - (le.x + re.x) / 2) / eyeSpan) * 90;
    const faceH = Math.abs(chin.y - top.y) || 0.001;
    const pitch = (((nose.y - (top.y + chin.y) / 2) / faceH) * 90);
    return {
      yaw: Math.max(-75, Math.min(75, yaw)),
      pitch: Math.max(-60, Math.min(60, pitch)),
      roll: Math.max(-60, Math.min(60, roll)),
    };
  }

  // ── Drawing ────────────────────────────────────────────────────────
  function mapPt(p, d) {
    return { x: d.x + p.x * d.w, y: d.y + p.y * d.h };
  }

  function strokeConnections(lm, d, connections, color, width, alpha) {
    if (!connections) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha == null ? 1 : alpha;
    ctx.lineWidth = Math.max(1, width);
    ctx.beginPath();
    const step = connections.length > 1200 ? 2 : 1;
    for (let i = 0; i < connections.length; i += step) {
      const c = connections[i];
      const a = lm[c.start], b = lm[c.end];
      if (!a || !b) continue;
      const pa = mapPt(a, d), pb = mapPt(b, d);
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawContour(lm, d, idxs, color, width) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1.2, width);
    ctx.beginPath();
    idxs.forEach((idx, i) => {
      const p = lm[idx];
      if (!p) return;
      const q = mapPt(p, d);
      i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }

  function drawDots(lm, d, unit) {
    ctx.save();
    for (let i = 0; i < lm.length; i++) {
      const p = lm[i];
      const depth = Math.max(-0.6, Math.min(0.8, p.z || 0));
      const r = Math.max(0.7, (1.05 - depth * 0.9) * unit);
      ctx.fillStyle = depth < -0.12
        ? 'rgba(28,92,158,0.9)'
        : 'rgba(46,116,192,0.72)';
      ctx.beginPath();
      ctx.arc(d.x + p.x * d.w, d.y + p.y * d.h, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawKinetic(lm, d) {
    if (!prevLm) return;
    ctx.save();
    ctx.strokeStyle = 'rgba(28,92,158,0.55)';
    ctx.lineWidth = Math.max(1, canvas.width / 700);
    ctx.beginPath();
    let active = 0;
    for (const i of KINETIC_IDX) {
      const a = prevLm[i], b = lm[i];
      if (!a || !b) continue;
      const x1 = d.x + a.x * d.w, y1 = d.y + a.y * d.h;
      const x2 = d.x + b.x * d.w, y2 = d.y + b.y * d.h;
      const dist = Math.hypot(x2 - x1, y2 - y1);
      if (dist < 0.45) continue;
      active++;
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();
    ctx.restore();
    activeVectors = active;
  }

  function drawGizmo(pose) {
    const R = Math.max(20, Math.min(34, canvas.width * 0.045));
    const cx = canvas.width - R - 20;
    const cy = canvas.height - R - 20;
    const yaw = (pose.yaw * Math.PI) / 180;
    const pitch = (pose.pitch * Math.PI) / 180;
    const roll = (pose.roll * Math.PI) / 180;

    const cyaw = Math.cos(yaw), syaw = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cr = Math.cos(roll), sr = Math.sin(roll);

    function project(x, y, z) {
      const x1 = x * cyaw - z * syaw;
      const z1 = x * syaw + z * cyaw;
      const y2 = y * cp - z1 * sp;
      const z2 = y * sp + z1 * cp;
      const x3 = x1 * cr - y2 * sr;
      const y3 = x1 * sr + y2 * cr;
      return { x: cx + x3 * R, y: cy + y3 * R, z: z2 };
    }

    const axes = [
      { v: [1, 0, 0], color: '#2e74c0', label: 'X' },
      { v: [0, 1, 0], color: '#1c5c9e', label: 'Y' },
      { v: [0, 0, 1], color: '#a06a06', label: 'Z' },
    ];

    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,0.86)';
    ctx.strokeStyle = 'rgba(195,217,238,0.9)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, R * 1.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    axes.forEach(a => {
      const p = project(a.v[0], a.v[1], a.v[2]);
      ctx.strokeStyle = a.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
      ctx.fillStyle = a.color;
      ctx.font = '600 8px Inter, sans-serif';
      ctx.fillText(a.label, p.x + 2, p.y - 2);
    });

    ctx.fillStyle = '#102a43';
    ctx.beginPath();
    ctx.arc(cx, cy, 2.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // ── Unified face handler ───────────────────────────────────────────
  function handleFace(lm, opts) {
    opts = opts || {};
    sizeCanvas();

    // FPS
    frames++;
    const now = performance.now();
    if (!fpsStamp) fpsStamp = now;
    if (now - fpsStamp >= 700) {
      if (fpsVal) fpsVal.textContent = Math.round((frames * 1000) / (now - fpsStamp));
      frames = 0;
      fpsStamp = now;
    }

    if (!video.videoWidth) return;
    const d = videoRect();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    try { ctx.drawImage(video, d.x, d.y, d.w, d.h); } catch (_) {}

    if (!lm) {
      if (faceChip) { faceChip.textContent = 'NO FACE'; faceChip.style.color = 'var(--red)'; }
      if (marVal) marVal.textContent = '—';
      if (vectorsEl) vectorsEl.textContent = '—';
      return;
    }

    const unit = canvas.width / 640;
    if (faceChip) { faceChip.textContent = 'FACE LOCKED'; faceChip.style.color = 'var(--indigo-light)'; }

    // 3D wireframe
    if (opts.connections) {
      strokeConnections(lm, d, opts.connections.tesselation, 'rgba(46,116,192,0.16)', unit * 0.7, 1);
      strokeConnections(lm, d, opts.connections.faceOval, 'rgba(28,92,158,0.7)', unit * 1.1, 1);
      strokeConnections(lm, d, opts.connections.lips, 'rgba(232,85,107,0.95)', unit * 1.25, 1);
      strokeConnections(lm, d, opts.connections.leftEye, 'rgba(46,116,192,0.95)', unit * 1.05, 1);
      strokeConnections(lm, d, opts.connections.rightEye, 'rgba(46,116,192,0.95)', unit * 1.05, 1);
      strokeConnections(lm, d, opts.connections.leftBrow, 'rgba(31,138,91,0.75)', unit, 1);
      strokeConnections(lm, d, opts.connections.rightBrow, 'rgba(31,138,91,0.75)', unit, 1);
    }
    drawDots(lm, d, unit);
    drawContour(lm, d, LIP_OUTER, 'rgba(232,85,107,0.9)', unit * 1.4);
    drawKinetic(lm, d);

    // Head pose
    const pose = (opts.matrix && poseFromMatrix(opts.matrix)) || poseFromLandmarks(lm);
    setPose('yaw', pose.yaw, 60);
    setPose('pitch', pose.pitch, 45);
    setPose('roll', pose.roll, 45);
    drawGizmo(pose);

    // Mouth + blendshapes
    const mar = computeMAR(lm, d.w, d.h);
    if (marVal) marVal.textContent = mar.toFixed(3);
    const bs = opts.blendshapes;
    const getBS = name => {
      if (!bs) return null;
      const c = bs.find(x => x.categoryName === name);
      return c ? c.score : null;
    };
    const jaw = getBS('jawOpen');
    if (jawEl) jawEl.textContent = jaw != null ? Math.round(jaw * 100) + '%' : Math.round(Math.min(1, mar / 0.5) * 100) + '%';
    if (blinkEl) {
      const bl = getBS('eyeBlinkLeft');
      const br = getBS('eyeBlinkRight');
      blinkEl.textContent = (bl != null && br != null)
        ? `${Math.round(bl * 100)}% / ${Math.round(br * 100)}%`
        : '—';
    }
    if (pointsEl) pointsEl.textContent = lm.length + ' pts';

    // Kinetic metrics
    let mean = 0, jitter = 0;
    if (prevLm) {
      let sum = 0, sumSq = 0, n = 0;
      for (const i of KINETIC_IDX) {
        const a = prevLm[i], b = lm[i];
        if (!a || !b) continue;
        const dist = Math.hypot((b.x - a.x) * d.w, (b.y - a.y) * d.h);
        sum += dist;
        sumSq += dist * dist;
        n++;
      }
      if (n) {
        mean = sum / n;
        jitter = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
      }
    }
    kineticEMA = kineticEMA * 0.82 + mean * 0.18;
    jitterEMA = jitterEMA * 0.9 + jitter * 0.1;
    if (kineticEl) kineticEl.textContent = kineticEMA.toFixed(2) + ' px';
    if (jitterEl) jitterEl.textContent = jitterEMA.toFixed(2) + ' px';
    if (vectorsEl) vectorsEl.textContent = activeVectors + ' active';

    // Keep previous landmarks for next frame (copy — MediaPipe reuses the array)
    prevLm = lm.map(p => ({ x: p.x, y: p.y, z: p.z }));
  }

  function computeMAR(lm, w, h) {
    const vert = Math.hypot((lm[L_UP].x - lm[L_DOWN].x) * w, (lm[L_UP].y - lm[L_DOWN].y) * h);
    const horz = Math.hypot((lm[L_LEFT].x - lm[L_RIGHT].x) * w, (lm[L_LEFT].y - lm[L_RIGHT].y) * h);
    return horz > 1 ? vert / horz : 0;
  }

  // ── Engines ────────────────────────────────────────────────────────
  async function loadTasksEngine() {
    const VERSION = '0.10.14';
    visionRef = await import(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/vision_bundle.mjs`);
    const fileset = await visionRef.FilesetResolver.forVisionTasks(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`
    );
    const options = {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    };
    try {
      landmarker = await visionRef.FaceLandmarker.createFromOptions(fileset, options);
    } catch (_) {
      options.baseOptions.delegate = 'CPU';
      landmarker = await visionRef.FaceLandmarker.createFromOptions(fileset, options);
    }
    engine = 'tasks';
    if (engineChip) engineChip.textContent = '3D · GPU/CPU';
  }

  function loadLegacyEngine() {
    return new Promise(resolve => {
      if (typeof window.FaceMesh === 'undefined') { resolve(false); return; }
      legacyMesh = new window.FaceMesh({
        locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`,
      });
      legacyMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      legacyMesh.onResults(res => {
        handleFace(res.multiFaceLandmarks && res.multiFaceLandmarks[0], {});
      });
      engine = 'legacy';
      if (engineChip) engineChip.textContent = '2D fallback';
      resolve(true);
    });
  }

  async function ensureEngine() {
    if (engineState === 'ready') return true;
    if (engineState === 'loading') return false;
    engineState = 'loading';
    setStatus('Loading 3D engine…', 'analyzing');
    if (meshEmpty) meshEmpty.textContent = 'Loading the 3D face-landmarker engine…';
    try {
      await loadTasksEngine();
      engineState = 'ready';
      if (meshEmpty) meshEmpty.textContent = 'Press play to start 3D landmark tracking on this video.';
      return true;
    } catch (_) {
      const ok = await loadLegacyEngine();
      if (ok) {
        engineState = 'ready';
        if (meshEmpty) meshEmpty.textContent = '3D engine unavailable — using the 2D face-mesh fallback. Press play to track.';
        return true;
      }
      engineState = 'failed';
      setStatus('Engine unavailable');
      if (meshEmpty) meshEmpty.textContent = 'The landmark engine could not load. Check your network connection — the rest of the analysis still works.';
      return false;
    }
  }

  // ── Detection loop ─────────────────────────────────────────────────
  async function tick() {
    if (!running) return;
    const now = performance.now();
    if (video && video.readyState >= 2 && !video.paused && !video.ended && now - lastSend > 62) {
      lastSend = now;
      if (engine === 'tasks' && landmarker) {
        try {
          const res = landmarker.detectForVideo(video, now);
          const lm = res.faceLandmarks && res.faceLandmarks[0];
          const bs = res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories;
          const mx = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0] && res.facialTransformationMatrixes[0].data;
          const F = visionRef && visionRef.FaceLandmarker;
          handleFace(lm, {
            blendshapes: bs,
            matrix: mx,
            connections: F ? {
              tesselation: F.FACE_LANDMARKS_TESSELATION,
              faceOval:    F.FACE_LANDMARKS_FACE_OVAL,
              lips:        F.FACE_LANDMARKS_LIPS,
              leftEye:     F.FACE_LANDMARKS_LEFT_EYE,
              rightEye:    F.FACE_LANDMARKS_RIGHT_EYE,
              leftBrow:    F.FACE_LANDMARKS_LEFT_EYEBROW,
              rightBrow:   F.FACE_LANDMARKS_RIGHT_EYEBROW,
            } : null,
          });
        } catch (_) {}
      } else if (engine === 'legacy' && legacyMesh) {
        try { await legacyMesh.send({ image: video }); } catch (_) {}
      }
    }
    requestAnimationFrame(tick);
  }

  async function startTracking() {
    if (!video || !video.videoWidth) return;
    const ok = await ensureEngine();
    if (!ok) return;
    if (meshEmpty) meshEmpty.style.display = 'none';
    if (meshHud) meshHud.style.display = 'flex';
    sizeCanvas();
    if (video.paused || video.ended) {
      paintFrame();
      setStatus('Ready');
      return;
    }
    setStatus('Tracking', 'analyzing');
    if (!running) {
      running = true;
      requestAnimationFrame(tick);
    }
  }

  function stopTracking() {
    running = false;
    setStatus('Paused');
  }

  // ── Video wiring ───────────────────────────────────────────────────
  const playBtn    = document.getElementById('mesh-playbtn');
  const playIcon   = document.getElementById('mesh-play-icon');
  const pauseIcon  = document.getElementById('mesh-pause-icon');

  function syncPlayBtn() {
    if (!playBtn) return;
    const playing = video && !video.paused && !video.ended;
    if (playIcon)  playIcon.style.display  = playing ? 'none'  : '';
    if (pauseIcon) pauseIcon.style.display = playing ? ''      : 'none';
  }

  if (playBtn && video) {
    playBtn.addEventListener('click', () => {
      if (video.paused || video.ended) {
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  }

  if (video) {
    video.addEventListener('loadeddata', () => { sizeCanvas(); paintFrame(); });
    video.addEventListener('play', () => {
      if (meshEmpty) meshEmpty.style.display = 'none';
      if (playBtn)   playBtn.style.display   = 'flex';
      syncPlayBtn();
      startTracking();
    });
    video.addEventListener('pause', () => {
      syncPlayBtn();
      if (video.currentTime < (video.duration || 0)) stopTracking();
    });
    video.addEventListener('ended', () => {
      syncPlayBtn();
      setStatus('Finished');
    });
    video.addEventListener('seeked', () => {
      syncPlayBtn();
      if (video.paused) { sizeCanvas(); paintFrame(); }
    });
  }

  if (stage && 'ResizeObserver' in window) {
    new ResizeObserver(() => {
      sizeCanvas();
      if (video && video.paused) paintFrame();
    }).observe(stage);
  } else {
    window.addEventListener('resize', () => {
      sizeCanvas();
      if (video && video.paused) paintFrame();
    });
  }

  // Click anywhere on the stage to toggle play / pause
  if (stage && video) {
    stage.addEventListener('click', e => {
      // Ignore clicks on the play button itself (it has its own handler)
      if (e.target === playBtn || playBtn?.contains(e.target)) return;
      if (!video.src) return;
      if (video.paused || video.ended) video.play().catch(() => {});
      else video.pause();
    });
  }

  // ── SHA-256 (chain of custody) ─────────────────────────────────────
  async function computeHash(file) {
    fileHash = null;
    if (!crypto || !crypto.subtle) return;
    if (file.size > 220 * 1024 * 1024) {
      fileHash = 'skipped (>220 MB)';
      window.ForensicDetail?.setHash(fileHash);
      return;
    }
    try {
      const buf = await file.arrayBuffer();
      const digest = await crypto.subtle.digest('SHA-256', buf);
      fileHash = Array.from(new Uint8Array(digest))
        .map(b => b.toString(16).padStart(2, '0')).join('');
      window.ForensicDetail?.setHash(fileHash);
    } catch (_) {
      fileHash = 'unavailable';
      window.ForensicDetail?.setHash(fileHash);
    }
  }

  // ── Public API ─────────────────────────────────────────────────────
  function setFile(file) {
    if (!video) return;
    clear();
    objectUrl = URL.createObjectURL(file);
    video.src = objectUrl;
    video.load();
    setStatus('Ready');
    if (meshEmpty) {
      meshEmpty.style.display = '';
      meshEmpty.textContent = 'Loading video…';
    }
    computeHash(file);

    // Warm the landmark engine, then auto-play when the video is ready
    ensureEngine().then(ok => {
      if (!ok) return;
      sizeCanvas();
      if (video.paused) paintFrame();
    });

    // Auto-play as soon as there is enough data to start
    video.addEventListener('canplay', function _autoPlay() {
      video.removeEventListener('canplay', _autoPlay);
      video.play().catch(() => {});   // browser may block — silently ignored
    }, { once: true });
  }

  function clear() {
    running = false;
    prevLm = null;
    kineticEMA = 0;
    jitterEMA = 0;
    activeVectors = 0;
    fileHash = null;
    window.ForensicDetail?.setHash('—');
    if (video) {
      video.pause();
      video.removeAttribute('src');
      video.load();
    }
    if (playBtn) { playBtn.style.display = 'none'; syncPlayBtn(); }
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (meshEmpty) { meshEmpty.style.display = ''; meshEmpty.textContent = 'Pick a video above to start landmark tracking. Press play — the 3D mesh follows the jaw, lips and eyes.'; }
    if (meshHud) meshHud.style.display = 'none';
    if (marVal) marVal.textContent = '—';
    if (jawEl) jawEl.textContent = '—';
    if (kineticEl) kineticEl.textContent = '—';
    if (jitterEl) jitterEl.textContent = '—';
    if (blinkEl) blinkEl.textContent = '—';
    if (pointsEl) pointsEl.textContent = '—';
    if (vectorsEl) vectorsEl.textContent = '—';
    Object.entries(poseEls).forEach(([, el]) => {
      if (el.val) el.val.textContent = '0°';
      if (el.bar) { el.bar.style.left = '50%'; el.bar.style.width = '0'; }
    });
    setStatus('Idle');
  }

  // ── Sync chart ─────────────────────────────────────────────────────
  function drawSyncChart(r) {
    const c = $('sync-chart');
    if (!c) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = c.getBoundingClientRect();
    const w = Math.max(200, Math.round(rect.width * dpr));
    const h = Math.round(120 * dpr);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const g = c.getContext('2d');
    g.clearRect(0, 0, w, h);

    const sig = r && r.signals;
    const tp = sig && sig.time_points;
    if (!tp || !tp.length) {
      g.fillStyle = '#9db0c2';
      g.font = `${11 * dpr}px Inter, sans-serif`;
      g.textAlign = 'center';
      g.fillText('Run an analysis to plot the audio / lip signals', w / 2, h / 2);
      g.textAlign = 'left';
      return;
    }

    const padL = 6 * dpr, padR = 6 * dpr, padT = 8 * dpr, padB = 16 * dpr;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // grid
    g.strokeStyle = 'rgba(195,217,238,0.75)';
    g.lineWidth = 1;
    for (let i = 0; i <= 2; i++) {
      const y = padT + (plotH / 2) * i;
      g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + plotW, y); g.stroke();
    }

    const norm = arr => {
      if (!arr || !arr.length) return [];
      let mn = Infinity, mx = -Infinity;
      arr.forEach(v => { if (v < mn) mn = v; if (v > mx) mx = v; });
      const span = (mx - mn) || 1;
      return arr.map(v => (v - mn) / span);
    };

    const audio = norm(sig.audio_energy);
    const lip = norm(sig.lip_mar);
    const curve = sig.sync_curve || [];
    const n = Math.max(audio.length, lip.length, curve.length);

    const line = (arr, color, width, alpha) => {
      if (!arr.length) return;
      g.save();
      g.strokeStyle = color;
      g.globalAlpha = alpha == null ? 1 : alpha;
      g.lineWidth = width;
      g.lineJoin = 'round';
      g.beginPath();
      for (let i = 0; i < arr.length; i++) {
        const x = padL + (i / Math.max(1, arr.length - 1)) * plotW;
        const y = padT + plotH - arr[i] * plotH;
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.stroke();
      g.restore();
    };

    // amber band for the sync curve
    line(curve, 'rgba(160,106,6,0.55)', 1.2 * dpr);
    line(audio, '#2e74c0', 1.6 * dpr);
    line(lip, '#1f8a5b', 1.6 * dpr);

    // axis labels
    g.fillStyle = '#9db0c2';
    g.font = `${9 * dpr}px Inter, sans-serif`;
    g.fillText('0s', padL, h - 4 * dpr);
    const dur = tp[tp.length - 1] || 0;
    const label = Math.round(dur) + 's';
    const tw = g.measureText(label).width;
    g.fillText(label, padL + plotW - tw, h - 4 * dpr);
  }

  function finding(pass) {
    return pass
      ? '<span class="finding pass">within</span>'
      : '<span class="finding fail">review</span>';
  }

  function fmtLag(ms) {
    if (ms == null) return '—';
    return ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms';
  }

  function renderSync(r) {
    const set = (id, text) => { const el = $(id); if (el) el.textContent = text; };
    const num = v => (v === null || v === undefined || Number.isNaN(Number(v))) ? null : Number(v);
    const sync = num(r.sync_score);
    const lag = num(r.audio_lip_delay_ms);
    const xcorr = num(r.xcorr_peak);
    const dtw = num(r.dtw_distance);
    const face = num(r.face_detection_rate);

    set('sp-sync', sync != null ? sync.toFixed(3) : '—');
    set('sp-lag', fmtLag(lag));
    set('sp-xcorr', xcorr != null ? xcorr.toFixed(3) : '—');
    set('sp-dtw', dtw != null ? dtw.toFixed(3) : '—');
    set('sp-face', face != null ? Math.round(face * 100) + '%' : '—');
    set('sp-rel', r.reliability_grade || '—');

    const fill = $('sp-fill');
    if (fill) fill.style.width = Math.round(Math.max(0, Math.min(1, sync || 0)) * 100) + '%';

    // Comparison table
    const tbody = $('sp-table');
    if (tbody) {
      const rows = [
        ['Lip–audio sync score', sync != null ? sync.toFixed(3) : '—', '≥ 0.62', sync != null && sync >= 0.62],
        ['Lip / speech offset', fmtLag(lag), '< 40 ms', lag != null && lag <= 40],
        ['Cross-correlation peak', xcorr != null ? xcorr.toFixed(3) : '—', '> 0.50', xcorr != null && xcorr >= 0.50],
        ['Dynamic time warping', dtw != null ? dtw.toFixed(3) : '—', '< 0.25', dtw != null && dtw < 0.25],
        ['Face tracking coverage', face != null ? Math.round(face * 100) + '%' : '—', '> 85%', face != null && face >= 0.85],
      ];
      tbody.innerHTML = rows.map(([name, obs, exp, pass]) =>
        `<tr><td>${name}</td><td>${obs}</td><td>${exp}</td><td>${finding(pass)}</td></tr>`
      ).join('');
    }

    // Interpretation
    const v = $('sp-verdict');
    if (v) {
      let cls = '', text = '';
      if (sync == null) {
        text = 'Awaiting analysis — run a video to populate the signal comparison.';
      } else if (sync >= 0.62 && (lag == null || lag <= 40)) {
        cls = 'good';
        text = `Signals align within biological tolerance. A lip/speech offset of ${fmtLag(lag || 0)} and a sync score of ${sync.toFixed(2)} are consistent with a genuine recording of a single speaker.`;
      } else if (sync >= 0.40) {
        cls = 'warn';
        text = `Partial alignment. The ${fmtLag(lag || 0)} offset and sync score of ${sync.toFixed(2)} are borderline — consistent with compression, dubbing or a lightly retimed track. Corroborate with the evidence board before concluding.`;
      } else {
        cls = 'bad';
        text = `Desynchronised. Lip motion trails speech by ${fmtLag(lag || 0)} while the sync score is only ${sync.toFixed(2)} — the mouth is not being driven by the same audio track, a strong indicator of face reenactment.`;
      }
      v.className = 'sync-verdict' + (cls ? ' ' + cls : '');
      v.textContent = text;
    }

    drawSyncChart(r);
  }

  function render(result) {
    lastResult = result || {};
    renderSync(lastResult);
  }

  window.addEventListener('resize', () => { if (lastResult) drawSyncChart(lastResult); });

  window.UploadDetail = {
    setFile,
    clear,
    render,
    getHash: () => fileHash,
  };
})();
