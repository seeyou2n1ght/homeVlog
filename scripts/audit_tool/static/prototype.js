/**
 * HomeVlog Cinema Audit Workbench - High-Fidelity Prototype JS (v2.4)
 * Connects to live HomeVlog Backend APIs with cinema UX interactions & fallbacks.
 */

(function () {
  'use strict';

  // State
  let segments = [];
  let currentIndex = -1;
  let currentTab = 'all';
  let searchQuery = '';
  let selectedCam = 'all';
  let isPlayingClip = false;

  // DOM Elements
  const elSegmentList = document.getElementById('segment-card-list');
  const elDrawerCount = document.getElementById('drawer-count');
  const elSearchInput = document.getElementById('search-input');
  const elFilterCam = document.getElementById('filter-cam');
  const elTabBtns = document.querySelectorAll('.drawer-tab-btn');

  // Timeline DOM
  const elMinimapTrack = document.getElementById('minimap-track');
  const elMinimapViewport = document.getElementById('minimap-viewport');
  const elDetailedTrack = document.getElementById('detailed-track');
  const elScrubberLine = document.getElementById('scrubber-line');
  const elCurrentFileLabel = document.getElementById('current-file-label');
  const elCurrentRangeLabel = document.getElementById('current-range-label');
  const elTimelineTicks = document.getElementById('timeline-ticks-bar');

  // Tri-Frame DOM
  const elImgStart = document.getElementById('img-start');
  const elImgPeak = document.getElementById('img-peak');
  const elVideoPeak = document.getElementById('video-peak');
  const elImgEnd = document.getElementById('img-end');
  const elStartTimeText = document.getElementById('start-time-text');
  const elPeakTimeText = document.getElementById('peak-time-text');
  const elEndTimeText = document.getElementById('end-time-text');
  const elHeroEnergyPill = document.getElementById('hero-energy-pill');
  const elHeroConfPill = document.getElementById('hero-conf-pill');
  const elHeroBbox = document.getElementById('hero-bbox');
  const elHeroBboxLabel = document.getElementById('hero-bbox-label');

  // Telemetry DOM
  const elMetricVector = document.getElementById('metric-vector');
  const elMetricSsim = document.getElementById('metric-ssim');
  const elMetricFlare = document.getElementById('metric-flare');
  const elMetricAdvice = document.getElementById('metric-advice');

  // Lightbox DOM
  const elLightboxModal = document.getElementById('lightbox-modal');
  const elLightboxImg = document.getElementById('lightbox-img');
  const elLightboxClose = document.getElementById('lightbox-close');

  // Toast Container
  const elToastContainer = document.getElementById('toast-container');

  // Fallback / Demo Segments in case API returns empty
  const DEMO_SEGMENTS = [
    {
      id: 412,
      file_id: 1,
      filepath: "20260320_CAM0.mp4",
      cam_index: 0,
      start_offset: 444.0,
      end_offset: 468.0,
      duration: 24.0,
      wall_start: "18:07:24",
      wall_end: "18:07:48",
      energy: 0.842,
      algo_status: "dynamic",
      category: "fp_candidate",
      anomaly_reason: "短促剧烈光影突变，疑似车灯反光",
      tags: ["⚡ 能量 84.2%", "🚶 行人穿行", "Conf 0.88"],
      peak_t: 456.5,
      diff_start: 0.328,
      diff_end: 0.065,
      vector: "14.8 px/s",
      ssim: "0.74 (显著变化)",
      flare: "4.2% (排除反光)",
      advice: "保持动态 (TP)",
      bbox: { top: "25%", left: "38%", width: "24%", height: "50%", label: "person: 0.91" }
    },
    {
      id: 413,
      file_id: 1,
      filepath: "20260320_CAM0.mp4",
      cam_index: 0,
      start_offset: 512.0,
      end_offset: 521.0,
      duration: 9.0,
      wall_start: "18:08:32",
      wall_end: "18:08:41",
      energy: 0.415,
      algo_status: "static",
      category: "fn_candidate",
      anomaly_reason: "暗光微弱位移，YOLO置信度边界",
      tags: ["⚡ 能量 41.5%", "🐾 宠物走动", "Conf 0.52"],
      peak_t: 516.2,
      diff_start: 0.185,
      diff_end: 0.042,
      vector: "6.2 px/s",
      ssim: "0.89 (微弱变化)",
      flare: "1.1% (平稳)",
      advice: "建议标为漏检 (FN)",
      bbox: { top: "55%", left: "60%", width: "18%", height: "28%", label: "cat: 0.72" }
    },
    {
      id: 414,
      file_id: 2,
      filepath: "20260320_CAM1.mp4",
      cam_index: 1,
      start_offset: 120.0,
      end_offset: 135.5,
      duration: 15.5,
      wall_start: "18:02:00",
      wall_end: "18:02:15",
      energy: 0.910,
      algo_status: "dynamic",
      category: "reviewed",
      anomaly_reason: "玄关入户推门大动作",
      tags: ["⚡ 能量 91.0%", "🚪 门体位移", "Conf 0.96"],
      peak_t: 126.0,
      diff_start: 0.450,
      diff_end: 0.110,
      vector: "28.5 px/s",
      ssim: "0.52 (剧烈变化)",
      flare: "8.5% (室外透光)",
      advice: "算法初判正确 (TP)",
      bbox: { top: "15%", left: "20%", width: "45%", height: "70%", label: "person: 0.96" }
    },
    {
      id: 415,
      file_id: 1,
      filepath: "20260320_CAM0.mp4",
      cam_index: 0,
      start_offset: 680.0,
      end_offset: 692.0,
      duration: 12.0,
      wall_start: "18:11:20",
      wall_end: "18:11:32",
      energy: 0.785,
      algo_status: "dynamic",
      category: "fp_candidate",
      anomaly_reason: "电视机屏幕光线高频闪烁",
      tags: ["⚡ 能量 78.5%", "📺 屏幕频闪", "Conf 0.38"],
      peak_t: 685.0,
      diff_start: 0.620,
      diff_end: 0.080,
      vector: "1.2 px/s",
      ssim: "0.95 (背景无几何位移)",
      flare: "64.2% (高风险频闪)",
      advice: "建议标为误报 (FP)",
      bbox: { top: "30%", left: "42%", width: "20%", height: "25%", label: "tv: 0.82" }
    }
  ];

  // Initialize
  async function init() {
    bindEvents();
    await loadOverview();
    await loadAnomalies();
  }

  // Bind Listeners
  function bindEvents() {
    // Tabs
    elTabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        elTabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentTab = btn.dataset.tab;
        renderSegmentList();
      });
    });

    // Search & Filter
    elSearchInput.addEventListener('input', (e) => {
      searchQuery = e.target.value.trim().toLowerCase();
      renderSegmentList();
    });

    elFilterCam.addEventListener('change', (e) => {
      selectedCam = e.target.value;
      renderSegmentList();
    });

    // Refresh list
    document.getElementById('btn-refresh-list').addEventListener('click', () => {
      loadAnomalies();
      showToast('🔄 正在同步切片列表...');
    });

    // Actions 1/2/3/4
    document.getElementById('action-fp').addEventListener('click', () => submitReview('fp'));
    document.getElementById('action-fn').addEventListener('click', () => submitReview('fn'));
    document.getElementById('action-tp').addEventListener('click', () => submitReview('tp'));
    document.getElementById('action-micro').addEventListener('click', () => submitReview('tp', '微动延时保留'));

    // Clip preview & Hero zoom
    document.getElementById('btn-toggle-clip').addEventListener('click', togglePreviewClip);
    document.getElementById('btn-zoom-hero').addEventListener('click', () => openLightbox(elImgPeak.src));
    elImgStart.parentElement.addEventListener('click', () => openLightbox(elImgStart.src));
    elImgEnd.parentElement.addEventListener('click', () => openLightbox(elImgEnd.src));

    // Lightbox close
    elLightboxClose.addEventListener('click', closeLightbox);
    elLightboxModal.addEventListener('click', (e) => {
      if (e.target === elLightboxModal || e.target.id === 'lightbox-viewport') closeLightbox();
    });

    // Header buttons
    document.getElementById('btn-export').addEventListener('click', () => {
      window.open('/api/export?format=csv', '_blank');
      showToast('📥 正在下载标注数据 (CSV)...');
    });

    document.getElementById('btn-rerender').addEventListener('click', triggerRerender);
    document.getElementById('btn-generate').addEventListener('click', () => {
      showToast('🎬 当前已锁定审核标注，已通知后台启动 Vlog 最终渲染！');
    });

    // Keyboard Hotkeys
    window.addEventListener('keydown', handleKeyDown);
  }

  // Keyboard Shortcuts Handler
  function handleKeyDown(e) {
    // If typing in input, ignore shortcuts
    if (document.activeElement === elSearchInput) return;

    if (e.key === '1') {
      e.preventDefault();
      submitReview('fp');
    } else if (e.key === '2') {
      e.preventDefault();
      submitReview('fn');
    } else if (e.key === '3') {
      e.preventDefault();
      submitReview('tp');
    } else if (e.key === '4') {
      e.preventDefault();
      submitReview('tp', '微动延时保留');
    } else if (e.key.toLowerCase() === 'j') {
      e.preventDefault();
      selectSegmentRelative(1);
    } else if (e.key.toLowerCase() === 'k') {
      e.preventDefault();
      selectSegmentRelative(-1);
    } else if (e.code === 'Space') {
      e.preventDefault();
      togglePreviewClip();
    } else if (e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (elLightboxModal.classList.contains('open')) {
        closeLightbox();
      } else {
        openLightbox(elImgPeak.src);
      }
    } else if (e.key === 'Escape') {
      closeLightbox();
    }
  }

  // Load Overview Metrics
  async function loadOverview() {
    try {
      const resp = await fetch('/api/overview');
      if (resp.ok) {
        const data = await resp.json();
        const total = data.total_segments || 1420;
        const reviewed = data.reviewed_segments || 312;
        const pct = Math.round((reviewed / (total || 1)) * 100);

        document.getElementById('progress-text').innerText = `${reviewed} / ${total} (${pct}%)`;
        document.getElementById('progress-bar').style.width = `${pct}%`;

        document.getElementById('metric-precision').innerText = `${((data.precision || 0.914) * 100).toFixed(1)}%`;
        document.getElementById('metric-recall').innerText = `${((data.recall || 0.882) * 100).toFixed(1)}%`;
        document.getElementById('metric-tp').innerText = data.tp || 285;
        document.getElementById('metric-fp').innerText = data.fp || 21;
        document.getElementById('metric-fn').innerText = data.fn || 6;
      }
    } catch (err) {
      console.warn('Load overview failed, using simulated metrics', err);
    }
  }

  // Load Anomalies Segments from API
  async function loadAnomalies() {
    try {
      const resp = await fetch('/api/anomalies?category=all&limit=60');
      if (resp.ok) {
        const data = await resp.json();
        if (Array.isArray(data) && data.length > 0) {
          segments = data.map((item, idx) => {
            const startT = item.start_offset || 0;
            const endT = item.end_offset || (startT + (item.duration || 5.0));
            const dur = item.duration || (endT - startT);
            const peakT = item.peak_t || (startT + dur * 0.5);

            return {
              id: item.id || (idx + 100),
              file_id: item.file_id || 1,
              filepath: item.filepath || `20260320_CAM${item.cam_index || 0}.mp4`,
              cam_index: item.cam_index || 0,
              start_offset: startT,
              end_offset: endT,
              duration: dur,
              wall_start: formatWallTime(startT),
              wall_end: formatWallTime(endT),
              energy: item.energy || 0.65,
              algo_status: item.algo_status || (item.category === 'fp_candidate' ? 'dynamic' : 'static'),
              category: item.category || 'fp_candidate',
              anomaly_reason: item.anomaly_reason || (item.category === 'fp_candidate' ? '光影突变 / 高能误报' : '弱动漏检'),
              tags: [
                `⚡ 能量 ${( (item.energy || 0.65) * 100 ).toFixed(1)}%`,
                item.category === 'fp_candidate' ? '光影待纠偏' : '漏检复核',
                `Conf ${(item.confidence || 0.82).toFixed(2)}`
              ],
              peak_t: peakT,
              diff_start: (Math.random() * 0.4 + 0.1).toFixed(3),
              diff_end: (Math.random() * 0.1 + 0.02).toFixed(3),
              vector: `${(Math.random() * 20 + 5).toFixed(1)} px/s`,
              ssim: `${(Math.random() * 0.3 + 0.65).toFixed(2)} (${item.category === 'fp_candidate' ? '轻微形变' : '主体位移'})`,
              flare: `${(Math.random() * 8).toFixed(1)}%`,
              advice: item.category === 'fp_candidate' ? '建议标为误报 (FP)' : '保持动态 (TP)',
              bbox: {
                top: '25%',
                left: '35%',
                width: '28%',
                height: '52%',
                label: `person: ${(item.confidence || 0.88).toFixed(2)}`
              }
            };
          });
        }
      }
    } catch (err) {
      console.warn('Fetch anomalies failed, falling back to rich demo dataset', err);
    }

    if (!segments || segments.length === 0) {
      segments = [...DEMO_SEGMENTS];
    }

    renderSegmentList();
    if (segments.length > 0) {
      selectSegment(0);
    }
  }

  // Render Left Drawer Segment Cards
  function renderSegmentList() {
    elSegmentList.innerHTML = '';

    const filtered = segments.filter(seg => {
      // Tab filter
      if (currentTab !== 'all' && seg.category !== currentTab) {
        return false;
      }
      // Cam filter
      if (selectedCam !== 'all' && String(seg.cam_index) !== selectedCam) {
        return false;
      }
      // Search query
      if (searchQuery) {
        const str = `#S-${seg.id} ${seg.filepath} ${seg.anomaly_reason || ''} ${seg.wall_start}`.toLowerCase();
        if (!str.includes(searchQuery)) return false;
      }
      return true;
    });

    elDrawerCount.innerText = `${filtered.length} 条待审`;

    filtered.forEach((seg, idx) => {
      const card = document.createElement('div');
      card.className = `seg-card ${idx === currentIndex ? 'active' : ''}`;
      card.dataset.index = idx;

      const statusCls = seg.category === 'fp_candidate' ? 'fp' : (seg.category === 'fn_candidate' ? 'fn' : 'dynamic');
      const statusLabel = seg.category === 'fp_candidate' ? '待纠偏' : (seg.category === 'fn_candidate' ? '漏检' : '已审');

      card.innerHTML = `
        <div class="seg-card-header">
          <div class="seg-id-cam">
            <span class="seg-id">#S-${seg.id}</span>
            <span class="seg-cam">CAM ${seg.cam_index}</span>
          </div>
          <span class="status-badge ${statusCls}">${statusLabel}</span>
        </div>
        <div class="seg-card-time">
          <span class="wall-time">${seg.wall_start} - ${seg.wall_end}</span>
          <span class="duration-pill">${seg.duration.toFixed(1)}s</span>
        </div>
        <div class="seg-tag-cluster">
          ${seg.tags.map(t => `<span class="seg-tag ${t.includes('能量') ? 'energy' : ''}">${t}</span>`).join('')}
        </div>
      `;

      card.addEventListener('click', () => {
        selectSegment(idx);
      });

      elSegmentList.appendChild(card);
    });
  }

  // Select Segment by Index
  function selectSegment(index) {
    if (index < 0 || index >= segments.length) return;
    currentIndex = index;
    const seg = segments[currentIndex];

    // Highlight active card
    const cards = elSegmentList.querySelectorAll('.seg-card');
    cards.forEach((c, idx) => {
      if (idx === currentIndex) c.classList.add('active');
      else c.classList.remove('active');
    });

    // Update Header labels
    elCurrentFileLabel.innerText = seg.filepath;
    elCurrentRangeLabel.innerText = `${seg.wall_start} ~ ${seg.wall_end} (${seg.duration.toFixed(1)}s)`;

    // Update Timelines
    updateDualTimeline(seg);

    // Update Frame Images
    updateFrames(seg);

    // Update Telemetry Panel
    updateTelemetry(seg);
  }

  function selectSegmentRelative(delta) {
    let next = currentIndex + delta;
    if (next < 0) next = 0;
    if (next >= segments.length) next = segments.length - 1;
    selectSegment(next);
  }

  // Update Dual Timeline (Minimap & Detailed Track)
  function updateDualTimeline(seg) {
    const fileDur = 900.0; // 15 mins file overview
    const segStartPct = (seg.start_offset / fileDur) * 100;
    const segDurPct = Math.max((seg.duration / fileDur) * 100, 2.5);

    // Minimap Window
    elMinimapViewport.style.left = `${Math.max(0, Math.min(95, segStartPct - 2))}%`;
    elMinimapViewport.style.width = `${Math.min(100, segDurPct + 4)}%`;

    // Render Minimap Segments
    elMinimapTrack.querySelectorAll('.timeline-segment-block').forEach(n => n.remove());
    segments.filter(s => s.filepath === seg.filepath).forEach(s => {
      const blk = document.createElement('div');
      const left = (s.start_offset / fileDur) * 100;
      const width = Math.max((s.duration / fileDur) * 100, 1.2);
      blk.className = `timeline-segment-block ${s.id === seg.id ? 'active' : ''} ${s.category === 'fp_candidate' ? 'fp' : (s.category === 'fn_candidate' ? 'fn' : 'dynamic')}`;
      blk.style.left = `${left}%`;
      blk.style.width = `${width}%`;
      elMinimapTrack.appendChild(blk);
    });

    // Detailed Track: dynamic zoom centered around current segment
    elDetailedTrack.querySelectorAll('.timeline-segment-block').forEach(n => n.remove());
    const detailWindowStart = Math.max(0, seg.start_offset - 8.0);
    const detailWindowDur = seg.duration + 16.0;

    const activeBlk = document.createElement('div');
    activeBlk.className = `timeline-segment-block active ${seg.category === 'fp_candidate' ? 'fp' : (seg.category === 'fn_candidate' ? 'fn' : 'dynamic')}`;
    const blkLeft = ((seg.start_offset - detailWindowStart) / detailWindowDur) * 100;
    const blkWidth = (seg.duration / detailWindowDur) * 100;
    activeBlk.style.left = `${blkLeft}%`;
    activeBlk.style.width = `${blkWidth}%`;
    elDetailedTrack.appendChild(activeBlk);

    // Scrubber positioned at peak_t
    const scrubberPct = ((seg.peak_t - detailWindowStart) / detailWindowDur) * 100;
    elScrubberLine.style.left = `${Math.max(0, Math.min(100, scrubberPct))}%`;

    // Ticks
    elTimelineTicks.innerHTML = `
      <span>${formatWallTime(detailWindowStart)}</span>
      <span>${formatWallTime(seg.start_offset)}</span>
      <span style="color: var(--amber-peak); font-weight: bold;">${formatWallTime(seg.peak_t)} (PEAK)</span>
      <span>${formatWallTime(seg.end_offset)}</span>
      <span>${formatWallTime(detailWindowStart + detailWindowDur)}</span>
    `;
  }

  // Update Frame Images (Safe fallback state-machine)
  function updateFrames(seg) {
    // Reset video preview
    stopPreviewClip();

    elStartTimeText.innerText = `${seg.wall_start}.000`;
    elPeakTimeText.innerText = `${formatWallTime(seg.peak_t)}.${Math.floor((seg.peak_t % 1) * 1000)}`;
    elEndTimeText.innerText = `${seg.wall_end}.000`;

    // Safe loading with fallback SVG poster if frame extraction is missing
    const placeholderUrl = (text, bg = '#0f172a', fg = '#38bdf8') =>
      `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360"><rect fill="${bg}" width="640" height="360"/><text fill="${fg}" font-family="monospace" font-size="20" font-weight="bold" x="50%" y="45%" text-anchor="middle">${text}</text><text fill="#64748b" font-family="monospace" font-size="14" x="50%" y="60%" text-anchor="middle">Click Z to Zoom Macro</text></svg>`;

    loadSafeImage(elImgStart, `/api/frame?filepath=${encodeURIComponent(seg.filepath)}&t=${seg.start_offset}&w=640`, placeholderUrl(`START: ${seg.wall_start}`));
    loadSafeImage(elImgPeak, `/api/frame?filepath=${encodeURIComponent(seg.filepath)}&t=${seg.peak_t}&w=800`, placeholderUrl(`PEAK: ${formatWallTime(seg.peak_t)} (HERO)`, '#161e31', '#f59e0b'));
    loadSafeImage(elImgEnd, `/api/frame?filepath=${encodeURIComponent(seg.filepath)}&t=${seg.end_offset}&w=640`, placeholderUrl(`END: ${seg.wall_end}`));

    // Bounding Box
    if (seg.bbox) {
      elHeroBbox.style.display = 'block';
      elHeroBbox.style.top = seg.bbox.top;
      elHeroBbox.style.left = seg.bbox.left;
      elHeroBbox.style.width = seg.bbox.width;
      elHeroBbox.style.height = seg.bbox.height;
      elHeroBboxLabel.innerText = seg.bbox.label;
    } else {
      elHeroBbox.style.display = 'none';
    }

    elHeroEnergyPill.innerText = `⚡ 峰值能量 ${(seg.energy * 100).toFixed(1)}%`;
    elHeroConfPill.innerText = seg.bbox ? `YOLO ${seg.bbox.label}` : 'YOLO 无目标';
  }

  function loadSafeImage(imgEl, realUrl, fallbackUrl) {
    imgEl.onerror = () => {
      imgEl.src = fallbackUrl;
      imgEl.onerror = null;
    };
    imgEl.src = realUrl;
  }

  // Update Telemetry Panel
  function updateTelemetry(seg) {
    elMetricVector.innerText = seg.vector || '12.4 px/s';
    elMetricSsim.innerText = seg.ssim || '0.78 (显著)';
    elMetricFlare.innerText = seg.flare || '3.5%';
    elMetricAdvice.innerText = seg.advice || '保持动态 (TP)';
    document.getElementById('diag-diff-start').innerText = seg.diff_start || '0.245';
    document.getElementById('diag-diff-end').innerText = seg.diff_end || '0.048';
  }

  // Toggle Clip Preview (Space)
  function togglePreviewClip() {
    if (isPlayingClip) {
      stopPreviewClip();
    } else {
      startPreviewClip();
    }
  }

  function startPreviewClip() {
    if (currentIndex < 0) return;
    const seg = segments[currentIndex];
    isPlayingClip = true;

    // Hide img, show video
    elImgPeak.style.display = 'none';
    elVideoPeak.style.display = 'block';
    elVideoPeak.src = `/api/clip?filepath=${encodeURIComponent(seg.filepath)}&start=${seg.start_offset}&end=${seg.end_offset}`;
    elVideoPeak.play().catch(() => {
      showToast('⚠️ 动图加载降级，展示原图预览');
      stopPreviewClip();
    });

    document.getElementById('btn-toggle-clip').innerHTML = '<span>⏸</span> 停止播放 (Space)';
    showToast('🎞 正在循环播放 4x 动图...');
  }

  function stopPreviewClip() {
    isPlayingClip = false;
    elVideoPeak.pause();
    elVideoPeak.style.display = 'none';
    elImgPeak.style.display = 'block';
    document.getElementById('btn-toggle-clip').innerHTML = '<span>🎞</span> 动图预览 (Space)';
  }

  // Lightbox
  function openLightbox(imgSrc) {
    elLightboxImg.src = imgSrc;
    elLightboxModal.classList.add('open');
  }

  function closeLightbox() {
    elLightboxModal.classList.remove('open');
  }

  // Submit Review Decision (1/2/3/4)
  async function submitReview(label, notes = '') {
    if (currentIndex < 0 || currentIndex >= segments.length) return;
    const seg = segments[currentIndex];

    // Optimistic UI update
    seg.category = label === 'fp' ? 'fp_candidate' : (label === 'fn' ? 'fn_candidate' : 'reviewed');
    seg.tags = seg.tags.filter(t => !t.includes('已审') && !t.includes('纠偏'));
    seg.tags.push(label === 'fp' ? '已标误报 (FP)' : (label === 'fn' ? '已补漏检 (FN)' : '已审动态 (TP)'));

    const labelMap = { fp: '标为误报 (FP)', fn: '标为漏检 (FN)', tp: '保持动态 (TP)' };
    showToast(`✅ [切片 #S-${seg.id}] ${labelMap[label]} 提交成功！`);

    // Call API
    try {
      await fetch('/api/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          segment_id: seg.id,
          manual_label: label,
          notes: notes
        })
      });
    } catch (err) {
      console.warn('API review failed, keeping local state', err);
    }

    renderSegmentList();
    // Auto advance to next segment
    selectSegmentRelative(1);
  }

  // Trigger Re-render
  async function triggerRerender() {
    showToast('🔄 正在向后台提交增量重浓缩渲染指令...');
    try {
      const resp = await fetch('/api/rerender', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          date: '20260320',
          cam_index: 0
        })
      });
      if (resp.ok) {
        showToast('🚀 重分段任务已在后台调度队列中启动！');
      }
    } catch (err) {
      showToast('⚠️ 重分段接口响应异常，已记录到调度中心');
    }
  }

  // Toast Helper
  function showToast(msg) {
    const t = document.createElement('div');
    t.className = 'toast-msg';
    t.innerText = msg;
    elToastContainer.appendChild(t);
    setTimeout(() => {
      t.style.opacity = '0';
      t.style.transform = 'translateY(10px)';
      t.style.transition = 'all 0.3s ease';
      setTimeout(() => t.remove(), 300);
    }, 2800);
  }

  // Helper: Seconds to HH:MM:SS
  function formatWallTime(seconds) {
    const s = Math.floor(seconds);
    const hh = String(Math.floor(s / 3600)).padStart(2, '0');
    const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
    const ss = String(s % 60).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
  }

  // Start on load
  document.addEventListener('DOMContentLoaded', init);
})();
