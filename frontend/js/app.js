/**
 * RAVIS — app.js
 * Handles: navigation, file upload, pipeline polling, results rendering,
 * demo presets, history table, report download.
 */

const API = 'http://localhost:5000';

// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  file:        null,
  jobId:       null,
  pollTimer:   null,
  result:      null,
  hash:        null,          // SHA-256 of the current file, when known
  history:     [],          // [{filename, verdict, confidence, sync, lag, date, sha256, result}]
  startTime:   null,
};

// ─── DOM ─────────────────────────────────────────────────────────────────────
const dom = {
  // Nav
  navLinks:       document.querySelectorAll('.nav-link'),
  pageTitle:      document.getElementById('page-title'),
  pageSub:        document.getElementById('page-sub'),

  // Status pill
  statusDot:      document.getElementById('status-dot'),
  statusText:     document.getElementById('status-text'),

  // Upload
  dropZone:       document.getElementById('drop-zone'),
  fileInput:      document.getElementById('file-input'),
  btnBrowse:      document.getElementById('btn-browse'),
  btnClear:       document.getElementById('btn-clear'),
  btnAnalyze:     document.getElementById('btn-analyze'),
  btnReset:       document.getElementById('btn-reset'),
  dropzoneIdle:   document.getElementById('dropzone-idle'),
  filePreview:    document.getElementById('file-preview'),
  fileName:       document.getElementById('file-name'),
  fileMeta:       document.getElementById('file-meta'),
  upPlayer:       document.getElementById('up-player'),
  upVideo:        document.getElementById('up-video'),
  uploadBadge:    document.getElementById('upload-badge'),

  // Progress
  progressWrap:   document.getElementById('progress-wrap'),
  progressStage:  document.getElementById('progress-stage'),
  progressPct:    document.getElementById('progress-pct'),
  progressFill:   document.getElementById('progress-fill'),
  pips: {
    1: document.getElementById('pip-1'),
    2: document.getElementById('pip-2'),
    3: document.getElementById('pip-3'),
    4: document.getElementById('pip-4'),
    5: document.getElementById('pip-5'),
    6: document.getElementById('pip-6'),
    7: document.getElementById('pip-7'),
  },

  // Results
  resultsEmpty:   document.getElementById('results-empty'),
  resultsContent: document.getElementById('results-content'),
  verdictBanner:  document.getElementById('verdict-banner'),
  verdictLabel:   document.getElementById('verdict-label'),
  verdictSub:     document.getElementById('verdict-sub'),
  confNumber:     document.getElementById('conf-number'),

  // Result summary
  mRel:   document.getElementById('m-rel'),

  // Explanation + timeline
  explText:       document.getElementById('expl-text'),
  timelineSection:document.getElementById('timeline-section'),
  timelineBarRow: document.getElementById('timeline-bar-row'),
  flaggedList:    document.getElementById('flagged-list'),

  // Actions
  btnDownload:    document.getElementById('btn-download-report'),
  btnNew:         document.getElementById('btn-new'),

  // History
  historyTbody:   document.getElementById('history-tbody'),
  btnClearHistory:document.getElementById('btn-clear-history'),
};

// ─── Pipeline stage → pip index ──────────────────────────────────────────────
const STAGE_PIP = {
  'Validating video':                       [1],
  'Extracting audio':                       [1,2],
  'Extracting video frames':                [1,2,3],
  'Detecting faces & lip landmarks':        [1,2,3,4],
  'Extracting audio features':              [1,2,3,4,5],
  'Extracting visual features':             [1,2,3,4,5],
  'Computing audio-visual synchronization': [1,2,3,4,5,6],
  'Running classifier':                     [1,2,3,4,5,6,7],
  'Generating explanation':                 [1,2,3,4,5,6,7],
  'Complete':                               [1,2,3,4,5,6,7],
};

// ─── Init ─────────────────────────────────────────────────────────────────────
function init() {
  setupNav();
  setupUpload();
  setupDemoButtons();
  setupResultButtons();
  setupHistory();
  setStatus('ready', 'Ready');
}

// ─── Navigation ───────────────────────────────────────────────────────────────
const PAGE_META = {
  upload:  { title: 'Upload Video',      sub: 'Upload a video to check if it\'s a deepfake' },
  live:    { title: 'Live Camera',       sub: 'Record yourself speaking and analyze in real-time' },
  history: { title: 'Analysis History', sub: 'All videos you\'ve analyzed this session' },
  about:   { title: 'About RAVIS',  sub: 'Features, how it works, and the tech stack' },
};

function setupNav() {
  dom.navLinks.forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const section = link.dataset.section;
      activateSection(section);
    });
  });
}

function activateSection(section) {
  // Update nav
  dom.navLinks.forEach(l => l.classList.toggle('active', l.dataset.section === section));

  // Update sections
  document.querySelectorAll('.content-section').forEach(s => {
    s.classList.toggle('active', s.id === `section-${section}`);
  });

  // Update page header
  const meta = PAGE_META[section] || {};
  if (dom.pageTitle) dom.pageTitle.textContent = meta.title || section;
  if (dom.pageSub)   dom.pageSub.textContent   = meta.sub   || '';
}

// ─── Status Pill ──────────────────────────────────────────────────────────────
function setStatus(type, text) {
  if (!dom.statusDot || !dom.statusText) return;
  dom.statusDot.className = 'status-dot ' + type;
  dom.statusText.textContent = text;
}

// ─── Upload & File Handling ───────────────────────────────────────────────────
function setupUpload() {
  dom.btnBrowse.addEventListener('click', e => { e.stopPropagation(); dom.fileInput.click(); });
  dom.fileInput.addEventListener('change', e => { if (e.target.files[0]) setFile(e.target.files[0]); });
  dom.btnClear.addEventListener('click', clearFile);
  dom.btnAnalyze.addEventListener('click', startAnalysis);
  dom.btnReset.addEventListener('click', resetAll);

  // Drag & drop
  dom.dropZone.addEventListener('click', () => { if (!state.file) dom.fileInput.click(); });
  dom.dropZone.addEventListener('dragover', e => { e.preventDefault(); dom.dropZone.classList.add('drag-over'); });
  dom.dropZone.addEventListener('dragleave', () => dom.dropZone.classList.remove('drag-over'));
  dom.dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dom.dropZone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });
  dom.dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); dom.fileInput.click(); }
  });
}

function setFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  const allowed = ['mp4','avi','mov','mkv','webm'];
  if (!allowed.includes(ext)) {
    alert(`File type .${ext} is not supported.\nPlease use: MP4, AVI, MOV, MKV, or WebM.`);
    return;
  }
  if (file.size > 500 * 1024 * 1024) {
    alert('File is too large. Maximum size is 500 MB.');
    return;
  }

  state.file = file;
  dom.fileName.textContent = file.name;
  dom.fileMeta.textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · ${ext.toUpperCase()}`;
  dom.dropzoneIdle.style.display = 'none';
  dom.dropZone.style.display     = 'none';
  dom.upPlayer.style.display     = 'block';
  dom.filePreview.style.display  = 'flex';
  dom.btnAnalyze.disabled = false;
  if (dom.uploadBadge) {
    dom.uploadBadge.textContent = `${ext.toUpperCase()} · ready`;
    dom.uploadBadge.classList.add('ready');
  }
  if (dom.upVideo) {
    try {
      if (dom.upVideo.src && dom.upVideo.src.startsWith('blob:')) {
        URL.revokeObjectURL(dom.upVideo.src);
      }
      const objUrl = URL.createObjectURL(file);
      dom.upVideo.src = objUrl;
      dom.upVideo.load();
      dom.upVideo.play().catch(() => {});
    } catch (e) {
      console.warn('Failed to load video preview:', e);
    }
  }
  window.UploadDetail?.setFile(file);
  setStatus('ready', 'File ready');
}

function clearFile() {
  state.file = null;
  dom.fileInput.value = '';
  dom.filePreview.style.display  = 'none';
  dom.dropzoneIdle.style.display = 'flex';
  dom.dropZone.style.display     = '';
  dom.upPlayer.style.display     = 'none';
  dom.btnAnalyze.disabled = true;
  if (dom.upVideo) {
    try {
      if (dom.upVideo.src && dom.upVideo.src.startsWith('blob:')) {
        URL.revokeObjectURL(dom.upVideo.src);
      }
      dom.upVideo.removeAttribute('src');
      dom.upVideo.load();
    } catch (e) {}
  }
  if (dom.uploadBadge) {
    dom.uploadBadge.textContent = 'No file';
    dom.uploadBadge.classList.remove('ready');
  }
  window.UploadDetail?.clear();
  setStatus('ready', 'Ready');
}

// ─── Analysis ─────────────────────────────────────────────────────────────────
async function startAnalysis() {
  if (!state.file) return;

  dom.btnAnalyze.disabled = true;
  dom.progressWrap.style.display = 'flex';
  dom.resultsEmpty.style.display = 'none';
  dom.resultsContent.style.display = 'none';
  state.startTime = Date.now();
  setStatus('running', 'Analyzing…');

  const form = new FormData();
  form.append('video', state.file);

  try {
    const res = await fetch(`${API}/api/analyze`, { method: 'POST', body: form });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `HTTP ${res.status}`);
    const data = await res.json();
    state.jobId = data.job_id;
    startPolling(data.job_id);
  } catch (err) {
    onError(
      'Could not reach the analysis server. Check that the backend is running, ' +
      'then try again — no result has been produced.'
    );
  }
}

function startPolling(jobId) {
  stopPolling();
  state.pollTimer = setInterval(async () => {
    try {
      const res  = await fetch(`${API}/api/status/${jobId}`);
      if (!res.ok) return;
      const data = await res.json();

      updateProgress(data.progress || 0, data.stage || 'Processing…');

      if (data.status === 'done') {
        stopPolling();
        onComplete(data.result);
      } else if (data.status === 'failed') {
        stopPolling();
        onError(data.error || 'Analysis failed. Please try again.');
      }
    } catch (e) { console.warn('Poll error:', e); }
  }, 900);
}

function stopPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

function updateProgress(pct, stage) {
  dom.progressFill.style.width    = pct + '%';
  dom.progressPct.textContent     = pct + '%';
  dom.progressStage.textContent   = stage;

  const donePips = STAGE_PIP[stage] || [];
  const lastPip  = donePips[donePips.length - 1];
  Object.entries(dom.pips).forEach(([num, el]) => {
    const n = parseInt(num);
    if (donePips.includes(n)) {
      el.className = 'pip ' + (n === lastPip && pct < 100 ? 'active' : 'done');
    } else {
      el.className = 'pip';
    }
  });
}

function onComplete(result) {
  if (state.startTime) result._elapsed_s = ((Date.now() - state.startTime) / 1000).toFixed(1);
  state.result = result;
  dom.progressWrap.style.display = 'none';
  // A demo/history entry has no real File behind it, so re-analysis must go
  // through the upload flow rather than sending a placeholder to the backend.
  dom.btnAnalyze.disabled = !(typeof File !== 'undefined' && state.file instanceof File);
  setStatus('done', `Done — ${result.verdict}`);
  renderResult(result, state.file?.name || 'video');
  state.hash = window.UploadDetail?.getHash?.() || null;
  saveHistory(state.file?.name || 'video', result, state.hash);
  renderHistory();
}

function onError(msg) {
  dom.progressWrap.style.display = 'none';
  dom.btnAnalyze.disabled = false;
  dom.resultsEmpty.style.display = 'flex';
  dom.resultsEmpty.innerHTML = `
    <p class="empty-title" style="color:var(--red)">Analysis failed</p>
    <p class="empty-sub">${escHtml(msg)}</p>`;
  setStatus('error', 'Failed');
}

// ─── Verdict presentation ─────────────────────────────────────────────────────
function verdictMeta(verdict) {
  const v = String(verdict || '').toUpperCase();
  if (v === 'DEEPFAKE') {
    return {
      cls: 'fake', label: 'DEEPFAKE', tag: 'FAKE',
      sub: 'Signs of manipulation detected in this video',
    };
  }
  if (v === 'REAL') {
    return {
      cls: 'real', label: 'AUTHENTIC', tag: 'REAL',
      sub: 'No manipulation detected in this video',
    };
  }
  return {
    cls: 'uncertain', label: 'INCONCLUSIVE', tag: 'N/A',
    sub: 'Evidence is not strong enough for a reliable verdict',
  };
}

// ─── Render Result ────────────────────────────────────────────────────────────
function renderResult(r, filename, container) {
  const meta       = verdictMeta(r.verdict);
  const isFake     = r.verdict === 'DEEPFAKE';
  const isUncertain= meta.cls === 'uncertain';
  const conf       = r.confidence        != null ? (r.confidence        * 100).toFixed(1) : '—';
  const syncScore  = r.sync_score        != null ? r.sync_score.toFixed(3)                : '—';
  const lagMs      = r.audio_lip_delay_ms != null
    ? (r.audio_lip_delay_ms >= 1000 ? (r.audio_lip_delay_ms / 1000).toFixed(2) + 's' : Math.round(r.audio_lip_delay_ms) + 'ms')
    : '—';
  const faceRate   = r.face_detection_rate!= null ? (r.face_detection_rate*100).toFixed(0)+'%' : '—';
  const relGrade   = r.reliability_grade  || '—';

  // Default: write into the upload results area
  if (!container) {
    dom.resultsContent.style.display = 'flex';
    dom.resultsEmpty.style.display   = 'none';

    // Verdict banner
    dom.verdictBanner.className = 'verdict-banner ' + meta.cls;
    dom.verdictLabel.textContent = meta.label;
    dom.verdictSub.textContent   = meta.sub;
    dom.confNumber.textContent   = isUncertain ? '—' : conf + '%';
    if (dom.mRel) dom.mRel.textContent = relGrade;

    // Explanation
    dom.explText.textContent = r.explanation?.summary || r.decision_reason ||
      (isFake
        ? `The model detected ${(r.deepfake_probability*100).toFixed(1)}% deepfake probability. Audio-lip sync score of ${syncScore} is below the natural threshold, and an audio-lip delay of ${lagMs} was detected.`
        : isUncertain
          ? 'The available measurements conflict or are too weak to support either an authentic or a manipulated verdict. No label is asserted.'
          : `The model found no significant signs of manipulation. Audio-lip movements are naturally synchronized (sync score: ${syncScore}, delay: ${lagMs}).`);

    // Timeline
    renderTimeline(r.timeline || [], r.suspicious_timestamps || []);

    // Dedicated A/V-sync + forensic evidence panels
    window.UploadDetail?.render(r);
    window.ForensicDetail?.render(r);
  } else {
    // Inject into a custom container (live section)
    container.innerHTML = buildResultHTML(r, isFake, conf, syncScore, lagMs, faceRate, relGrade);
  }
}

function renderTimeline(timeline, suspicious) {
  if (!dom.timelineBarRow) return;

  if (!timeline.length) {
    dom.timelineSection.style.display = 'none';
    return;
  }
  dom.timelineSection.style.display = 'block';

  // Bar row
  dom.timelineBarRow.innerHTML = '';
  timeline.forEach(seg => {
    const div = document.createElement('div');
    div.className = `tl-seg ${seg.status}`;
    div.title     = `${seg.timestamp} — Sync: ${(seg.sync_score * 100).toFixed(0)}%`;
    div.addEventListener('click', () => {
      // Could seek uploaded video — no-op for now
    });
    dom.timelineBarRow.appendChild(div);
  });

  // Flagged list
  dom.flaggedList.innerHTML = '';
  const bad = timeline.filter(s => s.status === 'bad');

  if (!bad.length) {
    dom.flaggedList.innerHTML = '<div class="flagged-ok">No suspicious moments found.</div>';
    return;
  }

  bad.slice(0, 8).forEach(seg => {
    const row = document.createElement('div');
    row.className = 'flag-row';
    row.innerHTML = `
      <span class="flag-ts">${seg.timestamp}</span>
      <span class="flag-desc">${seg.error_type || 'Lip-audio mismatch'}</span>
      <span class="flag-score">Sync: ${(seg.sync_score * 100).toFixed(0)}%</span>`;
    dom.flaggedList.appendChild(row);
  });

  if (bad.length > 8) {
    const more = document.createElement('div');
    more.className = 'flag-row';
    more.innerHTML = `<span class="flag-desc" style="color:var(--text3)">…and ${bad.length - 8} more suspicious moments</span>`;
    dom.flaggedList.appendChild(more);
  }
}

function buildResultHTML(r, isFake, conf, syncScore, lagMs, faceRate, relGrade) {
  const meta  = verdictMeta(r.verdict);
  const expl  = r.explanation?.summary || r.decision_reason ||
    (meta.cls === 'fake'
      ? `Deepfake detected (${(r.deepfake_probability*100).toFixed(1)}%). Sync score ${syncScore}, delay ${lagMs}.`
      : meta.cls === 'uncertain'
        ? 'The measurements do not support a reliable binary verdict. No label is asserted.'
        : `Authentic video confirmed. Sync score ${syncScore}, delay ${lagMs}.`);
  const confText = meta.cls === 'uncertain' ? '—' : conf + '%';

  return `
    <div class="verdict-banner ${meta.cls}">
      <div class="verdict-text">
        <div class="verdict-kicker">Overall result</div>
        <div class="verdict-label">${meta.label}</div>
        <div class="verdict-sub">${meta.sub}</div>
      </div>
      <div class="verdict-conf">
        <div class="conf-number">${confText}</div>
        <div class="conf-label">Confidence</div>
      </div>
    </div>
    <div class="live-result-body">
      <table class="core-table live-core-table">
        <thead><tr><th>Measurement</th><th>Observed</th></tr></thead>
        <tbody>
          <tr><td>Lip–audio sync</td><td>${syncScore}</td></tr>
          <tr><td>A/V delay</td><td>${lagMs}</td></tr>
          <tr><td>Face coverage</td><td>${faceRate}</td></tr>
          <tr><td>Reliability</td><td>${relGrade}</td></tr>
        </tbody>
      </table>
      <p class="expl-text">${escHtml(expl)}</p>
    </div>`;
}

// ─── Demo Fallback ────────────────────────────────────────────────────────────
function setupDemoButtons() {
  document.querySelectorAll('.btn-demo').forEach(btn => {
    btn.addEventListener('click', () => runDemo(btn.dataset.demo));
  });
}

function runDemo(type) {
  const isFake = type !== 'demo_authentic';
  const dur    = 30.0;

  stopPolling();
  state.startTime = Date.now();

  // Fake file object for display
  state.file   = { name: isFake ? 'deepfake_sample.mp4' : 'authentic_sample.mp4', size: 22 * 1024 * 1024 };
  state.jobId  = null;
  state.hash   = null;
  dom.fileName.textContent = state.file.name;
  dom.fileMeta.textContent = isFake ? '22.8 MB · MP4' : '18.4 MB · MP4';
  dom.dropzoneIdle.style.display = 'none';
  dom.dropZone.style.display     = 'none';
  if (dom.upPlayer) dom.upPlayer.style.display = 'none';
  dom.filePreview.style.display  = 'flex';
  if (dom.uploadBadge) {
    dom.uploadBadge.textContent = 'Demo · ready';
    dom.uploadBadge.classList.add('ready');
  }
  window.UploadDetail?.clear();
  dom.btnAnalyze.disabled = true;

  dom.progressWrap.style.display  = 'flex';
  dom.resultsEmpty.style.display  = 'none';
  dom.resultsContent.style.display= 'none';
  setStatus('running', 'Running demo…');

  // Simulate pipeline stages
  const stages = [
    { pct: 10, stage: 'Validating video',                       pip: 1 },
    { pct: 22, stage: 'Extracting audio',                       pip: 2 },
    { pct: 36, stage: 'Extracting video frames',                pip: 3 },
    { pct: 52, stage: 'Detecting faces & lip landmarks',        pip: 4 },
    { pct: 68, stage: 'Computing audio-visual synchronization', pip: 5 },
    { pct: 84, stage: 'Running classifier',                     pip: 6 },
    { pct: 96, stage: 'Generating explanation',                 pip: 7 },
    { pct: 100,stage: 'Complete',                               pip: 7 },
  ];

  let i = 0;
  const tick = () => {
    if (i >= stages.length) {
      updateProgress(100, 'Complete');
      Object.values(dom.pips).forEach(el => el.className = 'pip done');
      setTimeout(() => onComplete(generateDemoResult(isFake, dur)), 400);
      return;
    }
    const s = stages[i++];
    updateProgress(s.pct, s.stage);
    setTimeout(tick, 450 + Math.random() * 200);
  };
  tick();
}

function generateDemoResult(isFake, duration) {
  const syncScore    = isFake ? 0.19 + Math.random() * 0.12 : 0.72 + Math.random() * 0.14;
  const deepfakeProb = isFake ? 0.88 + Math.random() * 0.08 : 0.06 + Math.random() * 0.08;
  const lagMs        = isFake ? 120 + Math.floor(Math.random() * 200) : 12 + Math.floor(Math.random() * 25);
  const xcorr        = isFake ? 0.18 + Math.random() * 0.15 : 0.72 + Math.random() * 0.14;
  const faceRate     = isFake ? 0.72 + Math.random() * 0.18 : 0.92 + Math.random() * 0.07;

  const timeline = [];
  const nSec = Math.floor(duration);
  for (let s = 0; s < nSec; s++) {
    const score = isFake
      ? (s > 3 && s < 18 && Math.random() > 0.35) ? 0.1 + Math.random() * 0.28 : 0.45 + Math.random() * 0.35
      : 0.60 + Math.random() * 0.35;
    const status = score > 0.55 ? 'good' : score > 0.35 ? 'warn' : 'bad';
    timeline.push({
      second: s,
      timestamp: `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`,
      sync_score: parseFloat(score.toFixed(3)),
      status,
      error_type: status === 'bad' ? 'Lip-audio mismatch' : status === 'warn' ? 'Slight delay' : 'In sync',
      lag_ms: lagMs,
    });
  }

  return {
    verdict:              isFake ? 'DEEPFAKE' : 'REAL',    deepfake_probability: parseFloat(deepfakeProb.toFixed(4)),
    real_probability:     parseFloat((1 - deepfakeProb).toFixed(4)),
    confidence:           parseFloat(Math.max(deepfakeProb, 1 - deepfakeProb).toFixed(4)),
    sync_score:           parseFloat(syncScore.toFixed(4)),
    xcorr_peak:           parseFloat(xcorr.toFixed(4)),
    audio_lip_delay_ms:   lagMs,
    face_detection_rate:  parseFloat(faceRate.toFixed(3)),
    dtw_distance:         parseFloat((isFake ? 0.38 + Math.random()*0.2 : 0.08 + Math.random()*0.1).toFixed(4)),
    reliability_grade:    faceRate > 0.8 ? 'HIGH' : faceRate > 0.6 ? 'MEDIUM' : 'LOW',
    reliability_score:    parseFloat(faceRate.toFixed(3)),
    timeline,
    suspicious_timestamps: timeline.filter(t => t.status === 'bad').slice(0, 5),
    model_used:           'Demo',
    explanation: {
      summary: isFake
        ? `This video shows strong signs of deepfake manipulation. The lip movements are ${(lagMs > 80 ? `delayed by ${lagMs}ms compared to` : 'out of sync with')} the audio, and the sync score of ${syncScore.toFixed(2)} is well below the natural human threshold of 0.60. ${Math.round((1 - syncScore) * 100)}% of the signal points to manipulation.`
        : `This video appears authentic. The lip movements closely follow the audio with only ${lagMs}ms delay (within natural range of <40ms), and the sync score of ${syncScore.toFixed(2)} is above the threshold. No signs of deepfake manipulation were detected.`,
    },
    video_meta: { duration, fps: 25, width: 1920, height: 1080, file_size_mb: 22.8 },
    timestamp_generated: new Date().toLocaleString(),
    signals: (() => {
      const n = 120;
      const tp = [], au = [], lip = [], sc = [];
      const shift = isFake ? 0.55 : 0.03;
      for (let i = 0; i < n; i++) {
        const t = (i / (n - 1)) * (duration || 30);
        const a = Math.max(0, Math.min(1, 0.45 + 0.32 * Math.sin(t * 1.7) + 0.12 * Math.cos(t * 4.3)));
        const l = Math.max(0, Math.min(1, 0.45 + 0.30 * Math.sin((t - shift) * 1.7) + 0.10 * Math.cos((t - shift) * 4.3)));
        tp.push(+t.toFixed(2));
        au.push(+a.toFixed(3));
        lip.push(+l.toFixed(3));
        sc.push(+Math.max(0, Math.min(1, 1 - Math.abs(a - l) * (isFake ? 1.8 : 0.8))).toFixed(3));
      }
      return { time_points: tp, audio_energy: au, lip_mar: lip, sync_curve: sc };
    })(),
    attack_chips: isFake ? [
      { tag: `Lip-Audio Lag (${lagMs}ms)`, severity: 'critical' },
      { tag: `Low Sync Score (${syncScore.toFixed(2)})`, severity: 'critical' },
      { tag: 'Facial Inconsistency', severity: 'warning' },
    ] : [
      { tag: 'Natural Lip Sync', severity: 'authentic' },
      { tag: `Low Delay (${lagMs}ms)`, severity: 'authentic' },
    ],
  };
}

// ─── Result action buttons ─────────────────────────────────────────────────────
function setupResultButtons() {
  dom.btnNew?.addEventListener('click', resetAll);
  dom.btnDownload?.addEventListener('click', downloadReport);
}

function resetAll() {
  stopPolling();
  state.file   = null;
  state.jobId  = null;
  state.result = null;
  state.hash   = null;
  dom.fileInput.value                = '';
  dom.filePreview.style.display      = 'none';
  dom.dropzoneIdle.style.display     = '';
  if (dom.dropZone) dom.dropZone.style.display = '';
  if (dom.upPlayer) dom.upPlayer.style.display = 'none';
  if (dom.uploadBadge) {
    dom.uploadBadge.textContent = 'No file';
    dom.uploadBadge.classList.remove('ready');
  }
  window.UploadDetail?.clear();
  window.ForensicDetail?.clear();
  dom.btnAnalyze.disabled            = true;
  dom.progressWrap.style.display     = 'none';
  dom.resultsContent.style.display   = 'none';
  dom.resultsEmpty.innerHTML         = `
    <p class="empty-title">No results yet</p>
    <p class="empty-sub">Upload a video and click <strong>Analyze video</strong> to see the result.</p>`;
  dom.resultsEmpty.style.display     = 'flex';
  Object.values(dom.pips).forEach(el => el.className = 'pip');
  setStatus('ready', 'Ready');
  activateSection('upload');
}

async function downloadReport() {
  if (!state.result) return;
  const r      = state.result;
  const btn    = dom.btnDownload;
  const before = btn ? btn.innerHTML : '';
  if (btn) { btn.disabled = true; btn.textContent = 'Generating PDF…'; }
  setStatus('running', 'Generating PDF report…');

  try {
    const hash = window.UploadDetail?.getHash?.() || state.hash;
    const payload = {
      result: r,
      filename: state.file?.name || r.video_meta?.filename || 'video',
      elapsed: r._elapsed_s != null && Number.isFinite(Number(r._elapsed_s))
        ? Number(r._elapsed_s) : undefined,
    };
    if (typeof hash === 'string' && /^[a-f0-9]{64}$/i.test(hash)) payload.sha256 = hash;
    if (state.jobId) payload.job_id = state.jobId;

    const res = await fetch(`${API}/api/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.error || `HTTP ${res.status}`);
    }

    const blob = await res.blob();
    if (!blob.size) throw new Error('The server returned an empty PDF.');

    const cd = res.headers.get('Content-Disposition') || '';
    const m  = /filename\*?=(?:UTF-8'')?["']?([^"';]+)["']?/i.exec(cd);
    const name = m ? decodeURIComponent(m[1]) : `RAVIS_Report_${Date.now()}.pdf`;

    const url = URL.createObjectURL(blob);
    const a   = document.createElement('a');
    a.href     = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 15000);
    setStatus('done', 'PDF report downloaded');
  } catch (err) {
    console.error('PDF report failed:', err);
    downloadTextReport(r);
    setStatus('error', 'PDF unavailable — text report saved');
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = before; }
  }
}

// Last-resort fallback when the PDF service is unreachable.
function downloadTextReport(r) {
  const lines = [
    'RAVIS — Deepfake Detection Report',
    '='.repeat(48),
    `Generated  : ${new Date().toLocaleString()}`,
    `File       : ${state.file?.name || 'unknown'}`,
    '',
    'RESULT',
    '-'.repeat(40),
    `Verdict    : ${r.verdict}`,
    `Confidence : ${r.confidence != null ? (r.confidence*100).toFixed(2)+'%' : '—'}`,
    `Deepfake%  : ${r.deepfake_probability != null ? (r.deepfake_probability*100).toFixed(2)+'%' : '—'}`,
    `Decision   : ${r.decision_reason || '—'}`,
    '',
    'KEY METRICS',
    '-'.repeat(40),
    `Lip-Audio Sync  : ${r.sync_score != null ? r.sync_score.toFixed(4) : '—'}`,
    `Audio Delay     : ${r.audio_lip_delay_ms != null ? r.audio_lip_delay_ms+'ms' : '—'}`,
    `Face Detection  : ${r.face_detection_rate != null ? (r.face_detection_rate*100).toFixed(1)+'%' : '—'}`,
    `Reliability     : ${r.reliability_grade || '—'}`,
    `Cross-Corr Peak : ${r.xcorr_peak != null ? r.xcorr_peak.toFixed(4) : '—'}`,
    `DTW Distance    : ${r.dtw_distance != null ? r.dtw_distance.toFixed(5) : '—'}`,
    '',
    'EXPLANATION',
    '-'.repeat(40),
    r.explanation?.summary || r.decision_reason || '—',
    '',
    'SUSPICIOUS MOMENTS',
    '-'.repeat(40),
    ...(r.timeline?.filter(t=>t.status==='bad').map(t => `${t.timestamp} — Sync: ${(t.sync_score*100).toFixed(0)}% (${t.error_type})`) || ['None']),
    '',
    'VIDEO INFO',
    '-'.repeat(40),
    `Duration   : ${r.video_meta?.duration || '—'}s`,
    `Resolution : ${r.video_meta?.width||'—'}×${r.video_meta?.height||'—'}`,
    `FPS        : ${r.video_meta?.fps || '—'}`,
    `File Size  : ${r.video_meta?.file_size_mb || '—'} MB`,
    `Model Used : ${r.model_used || '—'}`,
  ];

  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `RAVIS_report_${Date.now()}.txt`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ─── History ──────────────────────────────────────────────────────────────────
function setupHistory() {
  dom.btnClearHistory?.addEventListener('click', () => {
    state.history = [];
    renderHistory();
  });

  // Load from localStorage
  try {
    const saved = JSON.parse(localStorage.getItem('dg_history') || '[]');
    state.history = saved.slice(0, 50);
  } catch (_) { state.history = []; }

  renderHistory();
}

function saveHistory(filename, result, sha256) {
  const hash = (typeof sha256 === 'string' && /^[a-f0-9]{64}$/i.test(sha256)) ? sha256 : null;
  const entry = {
    filename,
    verdict:    result.verdict,
    confidence: result.confidence != null ? (result.confidence*100).toFixed(1) : '—',
    sync:       result.sync_score  != null ? result.sync_score.toFixed(3)       : '—',
    lag:        result.audio_lip_delay_ms != null ? result.audio_lip_delay_ms+'ms' : '—',
    date:       new Date().toLocaleString('en-US', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' }),
    sha256:     hash,
    result,
  };
  state.history.unshift(entry);
  if (state.history.length > 50) state.history.pop();
  try { localStorage.setItem('dg_history', JSON.stringify(state.history)); } catch (_) {}
}

function renderHistory() {
  if (!dom.historyTbody) return;
  dom.historyTbody.innerHTML = '';

  if (!state.history.length) {
    dom.historyTbody.innerHTML = `<tr class="history-empty-row"><td colspan="8">No analyses yet. Upload a video to get started.</td></tr>`;
    return;
  }

  state.history.forEach((h, idx) => {
    const meta = verdictMeta(h.verdict);
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="td-muted">${idx + 1}</td>
      <td>${escHtml(h.filename)}</td>
      <td><span class="result-tag ${meta.cls}">${meta.tag}</span></td>
      <td class="td-muted">${meta.cls === 'uncertain' ? '—' : h.confidence + '%'}</td>
      <td class="td-muted">${h.sync}</td>
      <td class="td-muted">${h.lag}</td>
      <td class="td-muted">${h.date}</td>
      <td>
        <button class="btn-view-hist" data-idx="${idx}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
          View
        </button>
      </td>`;
    dom.historyTbody.appendChild(tr);
  });

  // Wire View buttons
  dom.historyTbody.querySelectorAll('.btn-view-hist').forEach(btn => {
    btn.addEventListener('click', () => {
      const entry = state.history[parseInt(btn.dataset.idx, 10)];
      if (entry?.result) showHistoryEntry(entry);
    });
  });
}

// Restore a saved analysis without re-running it or duplicating the history row.
function showHistoryEntry(entry) {
  stopPolling();
  state.result    = entry.result;
  state.file      = { name: entry.filename };
  state.jobId     = null;
  state.startTime = null;
  state.hash      = entry.sha256 || null;

  dom.progressWrap.style.display = 'none';
  dom.btnAnalyze.disabled        = true;

  // Make the upload card reflect the saved record rather than a previous file.
  dom.fileName.textContent = entry.filename;
  const vm  = entry.result.video_meta || {};
  const ext = (String(entry.filename).split('.').pop() || '').toUpperCase();
  const size = vm.file_size_mb != null ? `${vm.file_size_mb} MB` : '—';
  dom.fileMeta.textContent = `${size} · ${ext || '—'}`;
  dom.dropzoneIdle.style.display = 'none';
  if (dom.dropZone) dom.dropZone.style.display = 'none';
  if (dom.upPlayer) dom.upPlayer.style.display = 'none';
  dom.filePreview.style.display = 'flex';
  if (dom.uploadBadge) {
    dom.uploadBadge.textContent = 'History record';
    dom.uploadBadge.classList.add('ready');
  }

  window.UploadDetail?.clear();
  window.ForensicDetail?.clear();

  activateSection('upload');
  renderResult(entry.result, entry.filename);
  if (entry.sha256) window.ForensicDetail?.setHash(entry.sha256);
  else window.ForensicDetail?.setHash('not retained');
  setStatus('done', `History — ${entry.verdict}`);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ─── Expose helpers for realtime.js ───────────────────────────────────────────
window.DG = {
  renderResult,
  saveHistory,
  renderHistory,
  activateSection,
  setStatus,
  generateDemoResult,
  buildResultHTML,
  escHtml,
};

// ─── Boot ─────────────────────────────────────────────────────────────────────
init();
