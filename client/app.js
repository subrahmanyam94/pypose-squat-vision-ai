// PyPose Squat Vision AI - Client Application Logic (Vercel & Browser Ready)

// ── Backend URL Discovery & Configuration ─────────────────────────────────────
const urlParams = new URLSearchParams(window.location.search);
const defaultBase = (window.location.hostname === 'localhost' && window.location.port === '5000') 
  ? window.location.origin 
  : (localStorage.getItem('pypose_server_url') || urlParams.get('server') || 'http://localhost:5000');

let API_BASE = defaultBase.replace(/\/$/, '');

function updateApiBase(newUrl) {
  API_BASE = newUrl.trim().replace(/\/$/, '');
  localStorage.setItem('pypose_server_url', API_BASE);
  
  // Update video stream URL
  const img = document.getElementById('liveStreamImg');
  if (img) img.src = `${API_BASE}/video_feed`;
  
  // Update export links
  const csvLink = document.getElementById('linkExportCsv');
  if (csvLink) csvLink.href = `${API_BASE}/api/export_csv`;
  const jsonLink = document.getElementById('linkDumpJson');
  if (jsonLink) jsonLink.href = `${API_BASE}/api/dump_json`;
  
  console.log(`[PyPose] Connected backend URL: ${API_BASE}`);
}

// ── State variables ──────────────────────────────────────────────────────────
let isWireframe = true;
let isProtractor = true;
let isDepthLine = true;
let timerSeconds = 0;

// Session Timer
setInterval(() => {
  timerSeconds++;
  const hrs = String(Math.floor(timerSeconds / 3600)).padStart(2, '0');
  const mins = String(Math.floor((timerSeconds % 3600) / 60)).padStart(2, '0');
  const secs = String(timerSeconds % 60).padStart(2, '0');
  const el = document.getElementById('sessionTimer');
  if (el) el.textContent = `${hrs}:${mins}:${secs}`;
}, 1000);

// Tab Navigation
document.querySelectorAll('.nav-tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const targetTab = btn.getAttribute('data-tab');
    document.querySelectorAll('.nav-tab-btn').forEach(b => {
      b.classList.remove('bg-surface-container', 'text-primary', 'border-l-2', 'border-primary-container');
      b.classList.add('text-on-surface-variant');
    });
    btn.classList.add('bg-surface-container', 'text-primary', 'border-l-2', 'border-primary-container');
    btn.classList.remove('text-on-surface-variant');

    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.add('hidden'));
    const active = document.getElementById(`tab-${targetTab}`);
    if (active) active.classList.remove('hidden');
  });
});

// Control Buttons
document.getElementById('btnReset').addEventListener('click', () => {
  fetch(`${API_BASE}/api/reset`, { method: 'POST' }).catch(() => {});
});

document.getElementById('btnNextSet').addEventListener('click', () => {
  fetch(`${API_BASE}/api/next_set`, { method: 'POST' }).catch(() => {});
});

const skipRestBtn = document.getElementById('btnSkipRest');
if (skipRestBtn) {
  skipRestBtn.addEventListener('click', () => {
    fetch(`${API_BASE}/api/next_set`, { method: 'POST' }).catch(() => {});
  });
}

const resumeRestBtn = document.getElementById('btnResumeRest');
if (resumeRestBtn) {
  resumeRestBtn.addEventListener('click', () => {
    fetch(`${API_BASE}/api/resume_set`, { method: 'POST' }).catch(() => {});
  });
}

document.getElementById('btnTogglePause').addEventListener('click', () => {
  fetch(`${API_BASE}/api/toggle_pause`, { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      const lbl = document.getElementById('pauseLabel');
      const icn = document.getElementById('pauseIcon');
      if (d.state === 'paused') {
        lbl.textContent = 'RESUME';
        icn.textContent = 'play_arrow';
      } else {
        lbl.textContent = 'PAUSE';
        icn.textContent = 'pause';
      }
    })
    .catch(() => {});
});

document.getElementById('btnSnapshot').addEventListener('click', () => {
  fetch(`${API_BASE}/api/snapshot`, { method: 'POST' })
    .then(r => r.json())
    .then(d => alert('Snapshot captured: ' + d.file))
    .catch(() => alert('Could not capture snapshot. Check backend connection.'));
});

document.getElementById('btnFullscreen').addEventListener('click', () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen();
  } else {
    document.exitFullscreen();
  }
});

// Overlay Toggles
document.getElementById('btnToggleWireframe').addEventListener('click', () => {
  isWireframe = !isWireframe;
  fetch(`${API_BASE}/api/toggle_wireframe`, { method: 'POST' }).catch(() => {});
  document.getElementById('wireframeLabel').textContent = `Wireframe (${isWireframe ? 'ON' : 'OFF'})`;
});

document.getElementById('btnToggleProtractor').addEventListener('click', () => {
  isProtractor = !isProtractor;
  fetch(`${API_BASE}/api/toggle_protractor`, { method: 'POST' }).catch(() => {});
  document.getElementById('protractorLabel').textContent = `Protractor (${isProtractor ? 'ON' : 'OFF'})`;
});

document.getElementById('btnToggleDepth').addEventListener('click', () => {
  isDepthLine = !isDepthLine;
  fetch(`${API_BASE}/api/toggle_depth`, { method: 'POST' }).catch(() => {});
  document.getElementById('depthLineLabel').textContent = `Depth Line (${isDepthLine ? 'ON' : 'OFF'})`;
});

// Keyboard Shortcuts
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') {
    e.preventDefault();
    document.getElementById('btnTogglePause').click();
  } else if (e.code === 'KeyR') {
    document.getElementById('btnReset').click();
  } else if (e.code === 'KeyN') {
    document.getElementById('btnNextSet').click();
  } else if (e.code === 'KeyS') {
    document.getElementById('btnSnapshot').click();
  }
});

// Server URL Input change listener
const srvInput = document.getElementById('serverUrlInput');
if (srvInput) {
  srvInput.value = API_BASE;
  srvInput.addEventListener('change', () => {
    updateApiBase(srvInput.value);
  });
}

// ── Real-time Telemetry Polling (every 40ms = 25 Hz) ──────────────────────────
async function pollTelemetry() {
  try {
    const res = await fetch(`${API_BASE}/api/telemetry`, { mode: 'cors' });
    if (!res.ok) throw new Error('Network error');
    const d = await res.json();

    // Server connection badge
    const connDot = document.getElementById('serverConnDot');
    if (connDot) {
      connDot.className = 'w-2 h-2 rounded-full bg-primary-container animate-pulse';
    }

    // Header and top stats
    document.getElementById('hudFps').textContent = d.fps;
    document.getElementById('hudFpsBottom').textContent = d.fps;
    document.getElementById('hudLatency').textContent = d.inference_ms + 'ms';
    document.getElementById('hudScoreTop').textContent = Math.round(d.live_score);
    document.getElementById('hudChestHeader').textContent = d.chest_upright ? 'UPRIGHT' : 'LEANING';
    document.getElementById('hudChestHeader').className = d.chest_upright ? 'font-code-md text-primary-container font-bold' : 'font-code-md text-error font-bold';
    document.getElementById('hudChestStatusTop').textContent = d.chest_status;
    document.getElementById('hudChestBottom').textContent = d.chest_upright ? 'PASS' : 'FLAGGED';
    document.getElementById('hudFrameIdx').textContent = String(d.frame).padStart(5, '0');

    // 2-Minute Rest Interval Countdown & Modals
    if (d.is_resting) {
      const rem = Math.max(0, Math.round(d.rest_time_remaining || 0));
      const mins = String(Math.floor(rem / 60)).padStart(2, '0');
      const secs = String(rem % 60).padStart(2, '0');
      const timeStr = `${mins}:${secs}`;

      document.getElementById('restHeaderTimer').textContent = timeStr;
      document.getElementById('restBannerTimer').textContent = timeStr;
      document.getElementById('restHeaderBadge').classList.remove('hidden');
      document.getElementById('restOverlayBanner').classList.remove('hidden');
      
      const resumeBtn = document.getElementById('resumeRestBtnLabel');
      if (resumeBtn) resumeBtn.textContent = `CONTINUE SET ${String(d.current_set || 1).padStart(2, '0')}`;
      
      const skipBtn = document.getElementById('skipRestBtnLabel');
      if (skipBtn) skipBtn.textContent = `FINISH & START SET ${String((d.current_set || 1) + 1).padStart(2, '0')}`;
      
      const descEl = document.getElementById('restBannerDesc');
      if (descEl) {
        descEl.textContent = `Resting between squats. If you start squatting within 2 mins, Set ${String(d.current_set || 1).padStart(2, '0')} continues. After 2 mins, Set ${String((d.current_set || 1) + 1).padStart(2, '0')} starts automatically.`;
      }
      
      const nextBtnSpan = document.getElementById('btnNextSet').querySelector('span:last-child');
      if (nextBtnSpan) nextBtnSpan.textContent = 'FINISH SET (N)';
    } else {
      document.getElementById('restHeaderBadge').classList.add('hidden');
      document.getElementById('restOverlayBanner').classList.add('hidden');
      const nextBtnSpan = document.getElementById('btnNextSet').querySelector('span:last-child');
      if (nextBtnSpan) nextBtnSpan.textContent = 'NEXT SET (N)';
    }

    // Rep Counter & Stage
    const setEl = document.getElementById('liveSetCount');
    if (setEl && d.current_set) setEl.textContent = 'SET ' + String(d.current_set).padStart(2, '0');
    document.getElementById('liveRepCount').textContent = 'REP ' + String(d.reps).padStart(2, '0');
    document.getElementById('liveStageText').textContent = d.stage.toUpperCase() + ' (' + d.feedback + ')';

    // AR Callouts
    document.getElementById('arKneeAngle').textContent = d.knee_angle.toFixed(1) + '°';
    document.getElementById('arTorsoLean').textContent = d.torso_lean.toFixed(1) + '°';
    document.getElementById('arHipAngle').textContent = d.hip_angle.toFixed(1) + '°';
    document.getElementById('arLiveScore').textContent = Math.round(d.live_score) + ' / 100';

    // Chest pill
    const pillChest = document.getElementById('pillChestUpright');
    const pillChestText = document.getElementById('pillChestText');
    if (d.chest_upright) {
      pillChest.className = 'inline-flex items-center gap-1.5 bg-primary-container text-on-primary-container px-3 py-1 rounded font-label-caps text-xs uppercase font-bold shadow-md';
      pillChestText.textContent = `CHEST UPRIGHT • ${d.chest_status} (${d.torso_lean.toFixed(0)}°)`;
    } else {
      pillChest.className = 'inline-flex items-center gap-1.5 bg-error text-surface-container-lowest px-3 py-1 rounded font-label-caps text-xs uppercase font-bold shadow-md animate-pulse';
      pillChestText.textContent = `CHEST DROP FLAGGED • LEAN: ${d.torso_lean.toFixed(0)}°`;
    }

    // Depth pill
    document.getElementById('liveDepthDeg').textContent = d.knee_angle.toFixed(0) + '°';
    document.getElementById('liveDepthPct').textContent = d.depth_pct.toFixed(0) + '%';

    // Kinematics Rail Card
    document.getElementById('gaugeKneeText').textContent = d.knee_angle.toFixed(0) + '°';
    const ring = document.getElementById('svgProgressRing');
    const ringBadge = document.getElementById('ringPassBadge');
    const depthPct = Math.min(100, Math.max(0, d.depth_pct));
    ring.setAttribute('stroke-dasharray', `${depthPct}, 100`);

    if (d.knee_angle <= 95) {
      ring.setAttribute('class', 'text-primary-container');
      ringBadge.textContent = 'PASS';
      ringBadge.className = 'absolute inset-0 flex items-center justify-center font-code-md text-xs text-primary-container font-bold';
    } else {
      ring.setAttribute('class', 'text-secondary');
      ringBadge.textContent = d.stage.toUpperCase();
      ringBadge.className = 'absolute inset-0 flex items-center justify-center font-code-md text-[10px] text-on-surface font-bold';
    }

    document.getElementById('metricTorsoLean').textContent = d.torso_lean.toFixed(1) + '°';
    document.getElementById('metricTorsoStatus').textContent = d.chest_upright ? 'OPTIMAL' : 'LEANING';
    document.getElementById('metricHipAngle').textContent = d.hip_angle.toFixed(1) + '°';
    document.getElementById('metricScore').textContent = Math.round(d.live_score) + ' / 100';

    // Velocity & Bar path
    document.getElementById('vbtVelocity').textContent = d.velocity.toFixed(2);
    document.getElementById('vbtPeakVel').textContent = d.peak_velocity.toFixed(2);
    document.getElementById('barVel').style.width = Math.min(100, d.velocity * 50) + '%';
    document.getElementById('barPeakVel').style.width = Math.min(100, d.peak_velocity * 40) + '%';
    document.getElementById('vbtPower').textContent = d.power_watts + ' W';

    // Sagittal Drift
    document.getElementById('barDriftText').textContent = (d.bar_drift_cm > 0 ? '+' : '') + d.bar_drift_cm.toFixed(1) + ' cm';
    const driftPct = 50 + (d.bar_drift_cm / 10) * 50;
    document.getElementById('barDriftMarker').style.left = Math.max(5, Math.min(95, driftPct)) + '%';

    // Tab 2 Sync
    document.getElementById('tab2Reps').textContent = String(d.reps).padStart(2, '0');
    document.getElementById('tab2GoodRepsBadge').textContent = `${d.good_reps} / ${d.reps} VALID`;
    document.getElementById('tab2ChestScore').textContent = Math.round(d.live_score);
    document.getElementById('checkChestValue').textContent = d.torso_lean.toFixed(1) + '° LEAN';

    // Update Waveforms
    if (d.history && d.history.length > 2) {
      const pts = d.history;
      const w = 740;
      const h = 200;
      const n = pts.length;
      let kneeD = '';
      let hipD = '';
      let leanD = '';

      pts.forEach((p, idx) => {
        const x = 40 + (idx / (n - 1)) * w;
        const yKnee = 30 + ((180 - p.knee) / 120) * (180 - 30);
        const yHip = 30 + ((180 - p.hip) / 120) * (180 - 30);
        const yLean = 210 - (p.lean / 60) * 70;

        kneeD += (idx === 0 ? `M ${x} ${yKnee}` : ` L ${x} ${yKnee}`);
        hipD += (idx === 0 ? `M ${x} ${yHip}` : ` L ${x} ${yHip}`);
        leanD += (idx === 0 ? `M ${x} ${yLean}` : ` L ${x} ${yLean}`);
      });

      document.getElementById('wfKneePath').setAttribute('d', kneeD);
      document.getElementById('wfHipPath').setAttribute('d', hipD);
      document.getElementById('wfLeanPath').setAttribute('d', leanD);
    }

    // Rep velocity list
    if (d.rep_details && d.rep_details.length > 0) {
      const cont = document.getElementById('vbtRepBars');
      cont.innerHTML = d.rep_details.slice(-8).map(r => `
        <div class="flex items-center gap-2">
          <span class="w-12 text-on-surface-variant">REP ${String(r.rep).padStart(2, '0')}</span>
          <div class="flex-1 bg-surface-container-highest h-3 rounded overflow-hidden">
            <div class="bg-primary-container h-full" style="width: ${Math.min(100, (r.score || 80))}%"></div>
          </div>
          <span class="w-16 text-right font-bold text-primary">${Math.round(r.score)}% acc</span>
        </div>
      `).join('');
    }

    // Render Reps Per Set Bar Graph at bottom of HUD
    const setsCont = document.getElementById('repsPerSetContainer');
    if (setsCont) {
      const completed = d.completed_sets || [];
      const currentSetNum = d.current_set || 1;
      const currentRepsList = d.rep_details || [];

      const allSets = completed.map(cs => ({ ...cs, active: false }));
      allSets.push({
        set: currentSetNum,
        reps: d.reps || 0,
        good_reps: d.good_reps || 0,
        avg_score: currentRepsList.length > 0 ? Math.round(currentRepsList.reduce((acc, r) => acc + (r.score || 0), 0) / currentRepsList.length) : (d.live_score ? Math.round(d.live_score) : 0),
        rep_details: currentRepsList,
        active: true
      });

      const badgeEl = document.getElementById('repsPerSetBadge');
      if (badgeEl) {
        badgeEl.textContent = `SET ${String(currentSetNum).padStart(2, '0')} ACTIVE (${d.reps || 0} REPS)`;
      }

      setsCont.innerHTML = allSets.map(s => {
        const isActive = !!s.active;
        const reps = s.rep_details || [];
        const avgScore = s.avg_score || (reps.length > 0 ? Math.round(reps.reduce((a, b) => a + (b.score||0), 0) / reps.length) : 0);

        let barsHtml = '';
        if (reps.length === 0) {
          barsHtml = `
            <div class="h-24 flex flex-col items-center justify-center text-on-surface-variant text-xs font-code-md border border-dashed border-surface-container-highest rounded bg-surface-container-lowest/50">
              <span class="material-symbols-outlined text-[20px] mb-1 opacity-60 text-secondary">fitness_center</span>
              <span>Awaiting reps for Set ${String(s.set).padStart(2, '0')}...</span>
            </div>
          `;
        } else {
          const bars = reps.map((r, i) => {
            const sc = Math.round(r.score || 0);
            const hPct = Math.max(15, Math.min(100, sc));
            let colorClass = 'bg-primary-container text-on-primary-container';
            if (sc < 70) colorClass = 'bg-error text-surface-container-lowest';
            else if (sc < 85) colorClass = 'bg-secondary text-surface-container-lowest';

            return `
              <div class="flex-1 min-w-[28px] max-w-[48px] flex flex-col items-center justify-end h-full gap-1 group relative">
                <span class="text-[9px] font-code-md text-on-surface-variant group-hover:text-primary transition-colors">${sc}%</span>
                <div class="w-full ${colorClass} rounded-t transition-all duration-300 relative overflow-hidden shadow-sm hover:brightness-125" style="height: ${hPct}%;"></div>
                <span class="text-[9px] font-code-md text-secondary font-bold">R${r.rep || (i + 1)}</span>
              </div>
            `;
          }).join('');

          barsHtml = `
            <div class="h-24 bg-surface-container-lowest p-2 rounded border border-surface-container-highest flex items-end gap-2 overflow-x-auto">
              ${bars}
            </div>
          `;
        }

        const actionButtonHtml = isActive
          ? `<button type="button" class="btn-finish-set text-[10px] font-code-md bg-surface-container-highest hover:bg-secondary hover:text-surface-container-lowest text-secondary px-2 py-0.5 rounded transition-colors uppercase font-bold flex items-center gap-1 shadow-sm" data-set="${s.set}" title="Finish this set and advance to next set"><span class="material-symbols-outlined text-[12px]">skip_next</span>Finish Set</button>`
          : `<button type="button" class="btn-select-set text-[10px] font-code-md bg-surface-container-highest hover:bg-primary-container hover:text-on-primary-container text-primary px-2.5 py-0.5 rounded transition-colors uppercase font-bold flex items-center gap-1 shadow-sm" data-set="${s.set}" title="Continue adding reps to Set ${String(s.set).padStart(2, '0')}"><span class="material-symbols-outlined text-[12px]">replay</span>Continue Set ${String(s.set).padStart(2, '0')}</button>`;

        return `
          <div class="bg-surface-container p-4 rounded border ${isActive ? 'border-primary-container shadow-lg shadow-primary-container/10' : 'border-surface-container-high cursor-pointer hover:border-secondary/60'} transition-all flex flex-col gap-3">
            <div class="flex items-center justify-between pb-1 border-b border-surface-container-high">
              <div class="flex items-center gap-2">
                <span class="font-headline-sm text-sm font-bold text-primary">SET ${String(s.set).padStart(2, '0')}</span>
                ${isActive ? '<span class="inline-flex items-center gap-1 bg-primary-container text-on-primary-container px-1.5 py-0.5 rounded font-label-caps text-[9px] font-bold shadow-sm"><span class="w-1.5 h-1.5 rounded-full bg-on-primary-container animate-pulse"></span>ACTIVE</span>' : '<span class="font-label-caps text-[9px] text-on-surface-variant bg-surface-container-highest px-1.5 py-0.5 rounded">COMPLETED</span>'}
              </div>
              <div class="flex items-center gap-2">
                <span class="font-code-md text-xs ${s.good_reps > 0 ? 'text-primary-container font-bold' : 'text-on-surface-variant'}">${s.reps} REPS (${s.good_reps} VALID)</span>
                ${actionButtonHtml}
              </div>
            </div>

            ${barsHtml}

            <div class="flex items-center justify-between text-xs font-code-md text-on-surface-variant pt-1 border-t border-surface-container-highest">
              <span>AVG FORM: <strong class="text-primary font-bold">${avgScore}%</strong></span>
              <span>ACCURACY: <strong class="${s.reps > 0 && s.good_reps === s.reps ? 'text-primary-container' : 'text-secondary'} font-bold">${s.reps > 0 ? Math.round((s.good_reps / s.reps) * 100) : 100}%</strong></span>
            </div>
          </div>
        `;
      }).join('');

      setsCont.querySelectorAll('.btn-select-set').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const targetSet = parseInt(btn.getAttribute('data-set'), 10);
          fetch(`${API_BASE}/api/select_set`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ set: targetSet })
          }).catch(() => {});
        });
      });

      setsCont.querySelectorAll('.btn-finish-set').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          fetch(`${API_BASE}/api/next_set`, { method: 'POST' }).catch(() => {});
        });
      });
    }

  } catch (err) {
    const connDot = document.getElementById('serverConnDot');
    if (connDot) {
      connDot.className = 'w-2 h-2 rounded-full bg-error';
    }
  }
  setTimeout(pollTelemetry, 40);
}

// Initial connection
updateApiBase(API_BASE);
pollTelemetry();

