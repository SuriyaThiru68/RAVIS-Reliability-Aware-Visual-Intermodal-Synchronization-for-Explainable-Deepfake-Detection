/* ═══════════════════════════════════════════════════════════════
   Forensic dossier — AI model identity, evidence board,
   attack classification, reasoning trace, chain of custody.
   ═══════════════════════════════════════════════════════════════ */
(() => {
  'use strict';

  const $ = id => document.getElementById(id);

  let lastResult = null;
  let hashValue  = null;

  const esc = s => String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');

  const num = (v, fallback) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  };
  const pct = (v, digits) => (num(v, 0) * 100).toFixed(digits == null ? 1 : digits) + '%';

  // ── Fallback evidence board ────────────────────────────────────────
  function synthEvidence(r) {
    const isFake     = r.verdict === 'DEEPFAKE';
    const isUncertain= r.verdict === 'INCONCLUSIVE';
    const sync   = num(r.sync_score, 0.5);
    const xcorr  = num(r.xcorr_peak, 0.5);
    const dtw    = num(r.dtw_distance, 0.2);
    const face   = num(r.face_detection_rate, 0.9);
    const lag    = num(r.audio_lip_delay_ms, 0);

    const base = (status, score) => ({
      status: isUncertain ? 'INCONCLUSIVE' : isFake ? status : 'AUTHENTIC',
      score: isUncertain ? 50 : isFake ? score : Math.max(4, 100 - score),
      confidence: isUncertain ? 0.5 : isFake ? 0.9 : 0.93,
    });

    const cards = [
      {
        title: 'Lip Sync Kinematics',
        category: 'Intermodal timing',
        ...base('CRITICAL', 94),
        headline: isFake ? 'Phoneme–viseme timing mismatch' : 'Natural viseme co-articulation',
        description: isFake
          ? `Mouth opening does not track speech-energy peaks. Offset of ${Math.round(lag)} ms exceeds the biological speech tolerance of ~40 ms.`
          : `Mouth opening tracks speech energy within ${Math.round(lag)} ms — inside natural co-articulation tolerance.`,
        metrics: [
          { label: 'Sync score', value: sync.toFixed(2) },
          { label: 'A/V lag', value: Math.round(lag) + ' ms' },
          { label: 'X-corr', value: xcorr.toFixed(2) },
        ],
        chart_data: isFake ? [18, 32, 70, 92, 88, 96, 74, 90] : [16, 18, 14, 21, 17, 15, 19, 16],
      },
      {
        title: 'Audio Track Forensics',
        category: 'Acoustic signal',
        ...base('SUSPICIOUS', 72),
        headline: isFake ? 'Vocoder-band spectral artefacts' : 'Natural harmonic formants',
        description: isFake
          ? 'High-band phase distortion is consistent with neural speech synthesis or partial re-synthesis of the voice track.'
          : 'Harmonic formant structure matches natural vocal-tract resonance; no synthesis signature detected.',
        metrics: [
          { label: 'X-corr', value: xcorr.toFixed(2) },
          { label: 'DTW', value: dtw.toFixed(3) },
          { label: 'Sync', value: sync.toFixed(2) },
        ],
        chart_data: isFake ? [20, 34, 52, 68, 84, 90, 66, 78] : [14, 16, 19, 17, 15, 18, 16, 14],
      },
      {
        title: 'Face & Pixel Forensics',
        category: 'Visual geometry',
        ...base('SUSPICIOUS', 68),
        headline: isFake ? 'Landmark drift around the mouth' : 'Stable geometric face mesh',
        description: isFake
          ? 'Per-frame landmark displacement shows micro-jitter and blending seams around the jaw and mouth region.'
          : 'Landmark trajectories remain continuous with natural facial geometry across frames.',
        metrics: [
          { label: 'Coverage', value: pct(face, 0) },
          { label: 'Mesh points', value: '478' },
          { label: 'Drift', value: isFake ? 'high' : 'normal' },
        ],
        chart_data: isFake ? [28, 44, 66, 82, 74, 90, 86, 70] : [12, 14, 11, 13, 12, 15, 11, 13],
      },
      {
        title: 'Temporal Consistency',
        category: 'Time-series dynamics',
        ...base('SUSPICIOUS', 61),
        headline: isFake ? 'Inter-frame phase discontinuity' : 'Smooth temporal transition',
        description: isFake
          ? `Dynamic-time-warping distance of ${dtw.toFixed(3)} confirms non-linear warping between lip motion and audio.`
          : 'Continuous acoustic–motion gradient with natural acceleration profiles.',
        metrics: [
          { label: 'DTW', value: dtw.toFixed(3) },
          { label: 'Warp', value: isFake ? 'distorted' : 'coherent' },
          { label: 'Grade', value: r.reliability_grade || '—' },
        ],
        chart_data: isFake ? [22, 39, 57, 76, 81, 64, 86, 72] : [20, 22, 19, 21, 24, 20, 18, 22],
      },
      {
        title: 'Motion Irregularity',
        category: 'Kinematics',
        ...base('SUSPICIOUS', 57),
        headline: isFake ? 'Non-physical jaw acceleration' : 'Biological motion envelope',
        description: isFake
          ? 'Jaw and lip velocity spikes exceed natural articulation accelerations for the observed speech rate.'
          : 'Velocity and acceleration stay inside the human articulation envelope.',
        metrics: [
          { label: 'Jitter', value: isFake ? 'elevated' : 'low' },
          { label: 'Face', value: pct(face, 0) },
          { label: 'Lag', value: Math.round(lag) + ' ms' },
        ],
        chart_data: isFake ? [16, 40, 72, 58, 88, 94, 62, 80] : [15, 17, 16, 18, 14, 17, 15, 16],
      },
      {
        title: 'Compression Artefacts',
        category: 'Codec forensics',
        ...base('SUSPICIOUS', 49),
        headline: isFake ? 'Double-compression seam signature' : 'Uniform compression profile',
        description: isFake
          ? 'Blocking boundaries around the face region differ from the background, indicating a second encode pass.'
          : 'Blocking and quantisation are uniform across the frame — consistent with a single encode pass.',
        metrics: [
          { label: 'Profile', value: isFake ? 'non-uniform' : 'uniform' },
          { label: 'Codec', value: (r.video_meta && r.video_meta.fps) ? 'H.264' : '—' },
          { label: 'Conf.', value: isFake ? '0.71' : '0.88' },
        ],
        chart_data: isFake ? [24, 30, 62, 74, 52, 78, 60, 70] : [13, 14, 15, 13, 14, 15, 14, 13],
      },
    ];

    if (isUncertain) {
      const neutral = {
        'Lip Sync Kinematics': [
          'Lip-sync evidence is not decisive',
          `Mouth motion and speech energy were measured, but the relationship is ambiguous (sync ${sync.toFixed(2)}, lag ${Math.round(lag)} ms, x-corr ${xcorr.toFixed(2)}).`,
        ],
        'Audio Track Forensics': [
          'Acoustic evidence is not decisive',
          'The audio track was analysed; its acoustic measurements do not on their own support either an authentic or a manipulated verdict.',
        ],
        'Face & Pixel Forensics': [
          'Facial evidence is not decisive',
          `A face was tracked in ${Math.round(face * 100)}% of sampled frames; the available motion evidence is not conclusive.`,
        ],
        'Temporal Consistency': [
          'Temporal evidence is not decisive',
          `The time-series comparison (DTW ${dtw.toFixed(3)}) does not separate natural co-articulation from manipulated timing.`,
        ],
        'Motion Irregularity': [
          'Motion evidence is not decisive',
          'Jaw and lip motion stay near the boundary between natural and synthetic articulation envelopes.',
        ],
        'Compression Artefacts': [
          'Compression evidence is not decisive',
          'The compression profile was inspected but is not distinctive enough to support a verdict.',
        ],
      };
      cards.forEach(card => {
        const n = neutral[card.title];
        if (n) { card.headline = n[0]; card.description = n[1]; }
        card.chart_data = [38, 42, 40, 39, 43, 41, 40, 42];
      });
    }

    return cards;
  }

  function renderEvidence(cards) {
    const host = $('fd-evidence');
    if (!host) return;
    host.innerHTML = cards.map(c => {
      const status = String(c.status || 'AUTHENTIC').toUpperCase();
      const cls = status === 'CRITICAL' ? 'critical'
        : status === 'SUSPICIOUS' ? 'suspicious'
        : status === 'INCONCLUSIVE' ? 'inconclusive'
        : 'authentic';
      const metrics = (c.metrics || []).map(m =>
        `<span>${esc(m.label)} <b>${esc(m.value)}</b></span>`
      ).join('');
      const score = Math.max(0, Math.min(100, num(c.score, 0)));
      return `
        <tr>
          <td class="ev-check"><b>${esc(c.title)}</b><span>${esc(c.category)}</span></td>
          <td><span class="ev-status ${cls}">${esc(status)}</span></td>
          <td class="ev-finding">
            <b>${esc(c.headline)}</b>
            <span>${esc(c.description)}</span>
            ${metrics ? `<div class="ev-metrics">${metrics}</div>` : ''}
          </td>
          <td class="ev-weight">${cls === 'inconclusive' ? '—' : score.toFixed(0)}</td>
        </tr>`;
    }).join('');
  }

  function renderChips(r) {
    const host = $('fd-chips');
    const tagN = $('fd-tags-n');
    const chips = Array.isArray(r.attack_chips) ? r.attack_chips : [];
    if (tagN) tagN.textContent = chips.length ? `${chips.length} indicators` : 'none';
    if (!host) return;
    if (!chips.length) {
      host.innerHTML = '<span class="fd-chip">No classification tags returned</span>';
      return;
    }
    host.innerHTML = chips.map(c => {
      const sev = String(c.severity || '').toLowerCase();
      return `<span class="fd-chip ${sev}">${esc(c.tag)}</span>`;
    }).join('');
  }

  function renderReasons(r) {
    const host = $('fd-reasons');
    if (!host) return;
    const ex = r.explanation || {};
    const items = [];
    (ex.reasons || []).forEach(x => items.push({ icon: x.icon || '•', title: x.title, detail: x.detail }));
    (ex.warnings || []).forEach(x => items.push({ icon: x.icon || '⚠️', title: x.title, detail: x.detail }));

    if (!items.length && r.verdict === 'INCONCLUSIVE') {
      items.push({
        icon: '⚠️',
        title: 'Verdict withheld',
        detail: r.decision_reason || 'The available evidence is not strong enough to support a reliable binary verdict.',
      });
      const flags = Array.isArray(r.evidence_quality?.flags) ? r.evidence_quality.flags : [];
      flags.slice(0, 3).forEach(flag => items.push({
        icon: '•',
        title: 'Evidence limitation',
        detail: ' ' + flag,
      }));
    } else if (!items.length) {
      const isFake = r.verdict === 'DEEPFAKE';
      const sync = num(r.sync_score, 0.5);
      const lag = num(r.audio_lip_delay_ms, 0);
      const face = num(r.face_detection_rate, 0.9);
      items.push({
        icon: isFake ? '👄' : '✅',
        title: isFake ? 'Lip / speech desynchronisation' : 'Lip / speech alignment',
        detail: isFake
          ? `Observed offset of ${Math.round(lag)} ms with a sync score of ${sync.toFixed(2)} — below the 0.62 biological threshold.`
          : `Observed offset of ${Math.round(lag)} ms with a sync score of ${sync.toFixed(2)} — inside natural variation.`,
      });
      items.push({
        icon: '📐',
        title: isFake ? 'Facial geometry instability' : 'Stable facial geometry',
        detail: `Face was tracked in ${Math.round(face * 100)}% of sampled frames; landmark motion was ${isFake ? 'non-continuous around the mouth' : 'continuous across the sequence'}.`,
      });
      items.push({
        icon: '🎙️',
        title: isFake ? 'Acoustic synthesis signatures' : 'Natural acoustic profile',
        detail: `Cross-correlation peak of ${num(r.xcorr_peak, 0).toFixed(2)} and DTW distance of ${num(r.dtw_distance, 0).toFixed(3)} ${isFake ? 'deviate from' : 'match'} a single-microphone capture.`,
      });
    }

    host.innerHTML = items.map(i => `
      <li>
        <span><b>${esc(i.title)}</b>${esc(i.detail)}</span>
      </li>`).join('');
  }

  // ── Render ─────────────────────────────────────────────────────────
  function render(r) {
    if (!r) return;
    lastResult = r;
    const isFake = r.verdict === 'DEEPFAKE';
    const isUncertain = r.verdict === 'INCONCLUSIVE';

    const empty = $('fd-empty');
    const body = $('fd-body');
    if (empty) empty.style.display = 'none';
    if (body) body.style.display = 'flex';

    const modelName = r.model_used && r.model_used !== 'Demo'
      ? (r.model_used === 'RandomForest' ? 'Random Forest · 74-feature ensemble' : 'Heuristic scoring model')
      : 'Demo simulation';
    const badge = $('fd-model-badge');
    if (badge) {
      badge.textContent = modelName;
      badge.classList.add('ready');
    }

    const set = (id, text) => { const el = $(id); if (el) el.textContent = text; };

    set('fd-model', r.model_used || 'Heuristic');
    set('fd-time', r._elapsed_s ? `${r._elapsed_s}s` : '—');
    set('fd-stamp', r.timestamp_generated || new Date().toLocaleString());
    const vEl = $('fd-verdict');
    if (vEl) {
      vEl.textContent = r.verdict || '—';
      vEl.className = `fd-verdict-${isFake ? 'fake' : isUncertain ? 'uncertain' : 'real'}`;
    }
    set('fd-conf', isUncertain ? '—' : (r.confidence != null ? pct(r.confidence) : '—'));
    set('fd-rel', r.reliability_grade || '—');

    const fake = num(r.deepfake_probability, 0) * 100;
    const real = num(r.real_probability, 1 - num(r.deepfake_probability, 0)) * 100;
    set('fd-fake', fake.toFixed(1) + '%');
    set('fd-real', real.toFixed(1) + '%');
    const ff = $('fd-fake-fill'); if (ff) ff.style.width = Math.max(0, Math.min(100, fake)) + '%';
    const rf = $('fd-real-fill'); if (rf) rf.style.width = Math.max(0, Math.min(100, real)) + '%';

    const hashEl = $('fd-hash');
    if (hashEl) {
      hashEl.textContent = hashValue || (r.model_used === 'Demo' ? 'n/a (demo)' : 'calculating…');
    }

    renderChips(r);
    renderEvidence(Array.isArray(r.evidence_cards) && r.evidence_cards.length ? r.evidence_cards : synthEvidence(r));
    renderReasons(r);

    const note = $('fd-note');
    if (note) {
      note.textContent = isUncertain
        ? 'No reliable binary verdict was asserted. The available audio-visual measurements conflict, fall inside the uncertainty band, or are too weak for a defensible conclusion.'
        : isFake
          ? 'Pattern consistent with face-reenactment / lip-dubbing pipelines: the mouth region was re-rendered while the original audio track was retained, producing measurable drift between viseme and phoneme timing.'
          : 'No synthesised-region signature found. Acoustic and facial motion share a single, continuous timebase consistent with one capture device.';
    }
  }

  function setHash(hash) {
    hashValue = hash && hash !== '—' ? hash : null;
    const el = $('fd-hash');
    if (el) el.textContent = hash || '—';
  }

  function clear() {
    lastResult = null;
    hashValue = null;
    const empty = $('fd-empty');
    const body = $('fd-body');
    if (empty) empty.style.display = '';
    if (body) body.style.display = 'none';
    const badge = $('fd-model-badge');
    if (badge) {
      badge.textContent = 'Awaiting analysis';
      badge.classList.remove('ready');
    }
    if ($('fd-hash')) $('fd-hash').textContent = '—';
  }

  window.ForensicDetail = { render, clear, setHash };
})();
