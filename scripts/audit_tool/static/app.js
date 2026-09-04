// HomeVlog Studio Audit Console 2.0 Client Interaction Engine

// 全局应用状态
const state = {
  currentTab: 'anomalies', // 'anomalies' | 'tree'
  currentCategory: 'all',  // 'all' | 'fp_suspect' | 'fn_suspect' | 'jitter' | 'reviewed'
  sortMode: 'energy',      // 'energy' | 'time_asc' | 'time_desc' | 'duration'
  searchQuery: '',
  anomalies: [],
  filteredAnomalies: [],
  categoryCounts: {
    all: 0,
    fp_suspect: 0,
    fn_suspect: 0,
    jitter: 0,
    reviewed: 0
  },
  currentFile: null,
  currentSegments: [],
  selectedSegIndex: -1,
  sidebarCollapsed: false,
  
  // 双层时间轴状态
  timeline: {
    fileDuration: 300,
    fileOffsetSec: 0,      // 文件起始日内绝对秒数
    fileBaseClockDate: '', // 文件基准时间字符串
    viewStart: 0,          // Focus Track 可视窗口起始秒数 (相对视频 0)
    viewEnd: 300,          // Focus Track 可视窗口结束秒数
    isDraggingLens: false,
    dragStartX: 0,
    lensDragStartViewStart: 0,
    isPanningFocus: false,
    panStartX: 0,
    panStartViewStart: 0
  }
};

// DOM 初始化入口
document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  initLightbox();
  loadOverview();
  loadCategoryBadges();
  loadAnomalies();
  loadFileTree();
  loadArchiveStats();
});

// ================== 事件监听与初始化 ==================
function initEventListeners() {
  // 侧边栏折叠/展开
  const toggleBtn = document.getElementById('btn-sidebar-toggle');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      state.sidebarCollapsed = !state.sidebarCollapsed;
      const sidebar = document.getElementById('sidebar-panel');
      if (state.sidebarCollapsed) {
        sidebar.classList.add('collapsed');
        toggleBtn.innerText = '▶';
      } else {
        sidebar.classList.remove('collapsed');
        toggleBtn.innerText = '◀';
      }
    });
  }

  // 模式切换 Capsule
  const tabAno = document.getElementById('tab-anomalies');
  const tabTree = document.getElementById('tab-tree');
  if (tabAno) tabAno.addEventListener('click', () => switchTab('anomalies'));
  if (tabTree) tabTree.addEventListener('click', () => switchTab('tree'));

  // 错检分类胶囊点击切换
  document.querySelectorAll('.cat-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const cat = chip.getAttribute('data-category');
      switchCategory(cat);
    });
  });

  // 搜索过滤输入
  const filterInput = document.getElementById('filter-input');
  if (filterInput) {
    filterInput.addEventListener('input', (e) => {
      state.searchQuery = e.target.value.toLowerCase().trim();
      applyFilterAndSort();
    });
  }

  // 排序下拉框切换
  const sortSelect = document.getElementById('sort-select');
  if (sortSelect) {
    sortSelect.addEventListener('change', (e) => {
      state.sortMode = e.target.value;
      applyFilterAndSort();
    });
  }

  // 双层时间轴缩放控制按钮
  const btnZoomIn = document.getElementById('btn-zoom-in');
  const btnZoomOut = document.getElementById('btn-zoom-out');
  const btnZoomFocus = document.getElementById('btn-zoom-focus');
  const btnZoomReset = document.getElementById('btn-zoom-reset');

  if (btnZoomIn) btnZoomIn.addEventListener('click', () => zoomTimeline(0.6));
  if (btnZoomOut) btnZoomOut.addEventListener('click', () => zoomTimeline(1.5));
  if (btnZoomFocus) btnZoomFocus.addEventListener('click', () => focusCurrentSegment());
  if (btnZoomReset) btnZoomReset.addEventListener('click', () => resetTimelineZoom());

  // 双层时间轴交互事件：Minimap 镜头拖拽与点击
  initMinimapInteractions();
  // Focus Track 滚轮缩放与鼠标拖拽平移
  initFocusTrackInteractions();

  // 报表导出
  const btnExpCsv = document.getElementById('btn-export-csv');
  if (btnExpCsv) {
    btnExpCsv.addEventListener('click', () => {
      showToast('正在生成并下载 CSV 报表...', 'info');
      window.location.href = '/api/export?format=csv';
    });
  }
  const btnExpJson = document.getElementById('btn-export-json');
  if (btnExpJson) {
    btnExpJson.addEventListener('click', () => {
      showToast('正在生成并下载 JSON 报表...', 'info');
      window.location.href = '/api/export?format=json';
    });
  }

  // 工具按钮
  const btnYolo = document.getElementById('btn-yolo-enhance');
  if (btnYolo) btnYolo.addEventListener('click', runYoloEnhance);
  const btnClip = document.getElementById('btn-clip-preview');
  if (btnClip) btnClip.addEventListener('click', showClipPreview);

  // 切片上下切换按钮
  const btnPrevSeg = document.getElementById('btn-prev-seg');
  if (btnPrevSeg) btnPrevSeg.addEventListener('click', () => selectNextSegment(-1));
  const btnNextSeg = document.getElementById('btn-next-seg');
  if (btnNextSeg) btnNextSeg.addEventListener('click', () => selectNextSegment(1));

  // 打标按钮
  document.querySelectorAll('.decision-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const label = btn.getAttribute('data-label');
      submitCurrentReview(label);
    });
  });

  // 真实反馈帧归档库弹窗事件
  const btnOpenArchive = document.getElementById('btn-open-archive-modal');
  if (btnOpenArchive) btnOpenArchive.addEventListener('click', openArchiveModal);
  const btnArchiveClose = document.getElementById('btn-archive-close');
  if (btnArchiveClose) btnArchiveClose.addEventListener('click', closeArchiveModal);
  const btnArchiveModalClose = document.getElementById('btn-archive-modal-close');
  if (btnArchiveModalClose) btnArchiveModalClose.addEventListener('click', closeArchiveModal);
  const btnArchiveBatchRun = document.getElementById('btn-archive-batch-run');
  if (btnArchiveBatchRun) btnArchiveBatchRun.addEventListener('click', triggerBatchArchive);

  // 重渲染弹窗事件
  const btnOpenRerender = document.getElementById('btn-open-rerender');
  if (btnOpenRerender) btnOpenRerender.addEventListener('click', openRerenderModal);
  const btnRerenderClose = document.getElementById('btn-rerender-close');
  if (btnRerenderClose) btnRerenderClose.addEventListener('click', closeRerenderModal);
  const btnRerenderCancel = document.getElementById('btn-rerender-cancel');
  if (btnRerenderCancel) btnRerenderCancel.addEventListener('click', cancelOrCloseRerender);
  const btnRerenderStart = document.getElementById('btn-rerender-start');
  if (btnRerenderStart) btnRerenderStart.addEventListener('click', startRerender);
  const btnRerenderOpenDir = document.getElementById('btn-rerender-open-dir');
  if (btnRerenderOpenDir) btnRerenderOpenDir.addEventListener('click', openRerenderOutputDir);

  // 预览模态框关闭
  const btnModalClose = document.getElementById('btn-modal-close');
  if (btnModalClose) {
    btnModalClose.addEventListener('click', () => {
      document.getElementById('preview-modal').classList.add('hidden');
      document.getElementById('modal-preview-img').src = '';
    });
  }

  // 全局快捷键
  window.addEventListener('keydown', handleKeydown);
}

// 模式切换
function switchTab(tabName) {
  state.currentTab = tabName;
  const tabAno = document.getElementById('tab-anomalies');
  const tabTree = document.getElementById('tab-tree');
  const anoContent = document.getElementById('anomaly-content');
  const treeContent = document.getElementById('tree-content');
  const scopeDesc = document.getElementById('header-scope-desc');
  const catFilterBar = document.getElementById('category-filter-bar');

  if (tabName === 'anomalies') {
    tabAno.classList.add('active');
    tabTree.classList.remove('active');
    anoContent.classList.remove('hidden');
    treeContent.classList.add('hidden');
    if (catFilterBar) catFilterBar.classList.remove('hidden');
    if (scopeDesc) scopeDesc.innerText = '⚡ 疑难争议优先排查模式';
  } else {
    tabTree.classList.add('active');
    tabAno.classList.remove('active');
    treeContent.classList.remove('hidden');
    anoContent.classList.add('hidden');
    if (catFilterBar) catFilterBar.classList.add('hidden');
    if (scopeDesc) scopeDesc.innerText = '📁 全量素材文件浏览模式';
  }
}

// 分类切换
function switchCategory(cat) {
  state.currentCategory = cat;
  document.querySelectorAll('.cat-chip').forEach(c => {
    if (c.getAttribute('data-category') === cat) {
      c.classList.add('active');
    } else {
      c.classList.remove('active');
    }
  });
  loadAnomalies();
}

// 键盘快捷键监听
function handleKeydown(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

  // 若高清细节放大视窗开启，优先响应视窗内缩放与关闭
  if (lightboxState && lightboxState.isOpen) {
    if (e.key === 'Escape') {
      closeLightbox();
      return;
    } else if (e.key === '+' || e.key === '=') {
      zoomLightbox(1.25);
      return;
    } else if (e.key === '-' || e.key === '_') {
      zoomLightbox(0.8);
      return;
    } else if (e.key === '0') {
      resetLightboxZoom();
      return;
    } else if (e.key === '1') {
      setLightboxActualSize();
      return;
    }
  }

  if (e.key === '1') {
    submitCurrentReview('CONFIRMED_MOTION');
  } else if (e.key === '2') {
    submitCurrentReview('FALSE_ALARM');
  } else if (e.key === '3') {
    submitCurrentReview('MISSED_MOTION');
  } else if (e.key === '4') {
    submitCurrentReview('CONFIRMED_STATIC');
  } else if (e.key === 'j' || e.key === 'ArrowDown') {
    selectNextSegment(1);
  } else if (e.key === 'k' || e.key === 'ArrowUp') {
    selectNextSegment(-1);
  } else if (e.key === 'y' || e.key === 'Y') {
    runYoloEnhance();
  } else if (e.key === ' ') {
    e.preventDefault();
    showClipPreview();
  } else if (e.key === 'Escape') {
    const pModal = document.getElementById('preview-modal');
    if (pModal) pModal.classList.add('hidden');
    const rModal = document.getElementById('rerender-modal');
    if (rModal && !rModal.classList.contains('hidden')) closeRerenderModal();
    const aModal = document.getElementById('archive-modal');
    if (aModal && !aModal.classList.contains('hidden')) closeArchiveModal();
    if (lightboxState && lightboxState.isOpen) closeLightbox();
  }
}

// Toast 提示条
function showToast(msg, type = 'success') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = 'toast';
  const icon = type === 'success' ? '✅' : (type === 'error' ? '❌' : 'ℹ️');
  toast.innerHTML = `<span>${icon}</span><span>${msg}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, 2500);
}

// ================== 全局看板与各分类数量徽标 ==================
async function loadOverview() {
  try {
    const res = await fetch('/api/overview');
    const data = await res.json();
    const totalSegsEl = document.getElementById('metric-total-segs');
    if (totalSegsEl) totalSegsEl.innerText = `/ ${data.total_segments}`;
    
    const revSegsEl = document.getElementById('metric-reviewed-segs');
    if (revSegsEl && revSegsEl.childNodes.length > 0) {
      revSegsEl.childNodes[0].nodeValue = `${data.reviewed_segments} `;
    }
    
    const tpEl = document.getElementById('metric-tp');
    if (tpEl) tpEl.innerText = data.labels.tp;
    const fpEl = document.getElementById('metric-fp');
    if (fpEl) fpEl.innerText = data.labels.fp;
    const fnEl = document.getElementById('metric-fn');
    if (fnEl) fnEl.innerText = data.labels.fn;
    const precRecEl = document.getElementById('metric-prec-rec');
    if (precRecEl) precRecEl.innerText = `${data.metrics.precision}% / ${data.metrics.recall}%`;

    const pct = data.total_segments > 0 ? (data.reviewed_segments / data.total_segments) * 100 : 0;
    const barEl = document.getElementById('metric-progress-bar');
    if (barEl) barEl.style.width = `${pct.toFixed(1)}%`;
  } catch (err) {
    console.error('Failed to load overview:', err);
  }
}

// 统计并刷新分类徽标
async function loadCategoryBadges() {
  try {
    const res = await fetch('/api/anomalies?category=all&limit=200');
    const allList = await res.json();
    if (!Array.isArray(allList)) return;

    let fpCount = 0;
    let fnCount = 0;
    let jitterCount = 0;
    let reviewedCount = 0;

    allList.forEach(item => {
      if (item.manual_label) {
        reviewedCount++;
      } else if (item.state === 'DYNAMIC' && (item.avg_confidence === 0 || item.max_energy < 1.0)) {
        fpCount++;
      } else if (item.state === 'STATIC' && item.max_energy >= 1.5) {
        fnCount++;
      } else if (item.duration < 3.0) {
        jitterCount++;
      }
    });

    state.categoryCounts = {
      all: allList.length,
      fp_suspect: fpCount,
      fn_suspect: fnCount,
      jitter: jitterCount,
      reviewed: reviewedCount
    };

    updateBadgeElement('badge-cat-all', allList.length);
    updateBadgeElement('badge-cat-fp', fpCount);
    updateBadgeElement('badge-cat-fn', fnCount);
    updateBadgeElement('badge-cat-jitter', jitterCount);
    updateBadgeElement('badge-cat-reviewed', reviewedCount);
  } catch (e) {
    console.warn('Failed to load category badges:', e);
  }
}

function updateBadgeElement(id, count) {
  const el = document.getElementById(id);
  if (el) el.innerText = count.toString();
}

// ================== 疑难样本列表与过滤排序 ==================
async function loadAnomalies() {
  const container = document.getElementById('anomaly-items');
  if (container) {
    container.innerHTML = '<div class="state-loader"><div class="spin-ring"></div> 正在挖掘分类争议样本...</div>';
  }

  try {
    const url = `/api/anomalies?category=${state.currentCategory}&limit=100`;
    const res = await fetch(url);
    const list = await res.json();
    state.anomalies = list || [];
    
    const countEl = document.getElementById('anomaly-count');
    if (countEl) countEl.innerText = state.anomalies.length;

    applyFilterAndSort();

    // 默认聚焦第一条
    if (state.filteredAnomalies.length > 0 && container) {
      const firstCard = container.querySelector('.anomaly-card');
      if (firstCard) firstCard.click();
    }
  } catch (err) {
    console.error('Failed to load anomalies:', err);
    if (container) container.innerHTML = '<div class="state-loader error">加载争议样本失败，请检查服务状态</div>';
  }
}

function applyFilterAndSort() {
  let list = [...state.anomalies];

  // 1. 文本搜索过滤
  if (state.searchQuery) {
    const q = state.searchQuery;
    list = list.filter(item => {
      const fn = (item.filepath || '').split('\\').pop().split('/').pop().toLowerCase();
      const eng = (item.max_energy || 0).toString();
      const st = (item.state || '').toLowerCase();
      const reason = (item.reason_desc || '').toLowerCase();
      return fn.includes(q) || eng.includes(q) || st.includes(q) || reason.includes(q);
    });
  }

  // 2. 排序引擎
  if (state.sortMode === 'energy') {
    list.sort((a, b) => (b.max_energy || 0) - (a.max_energy || 0));
  } else if (state.sortMode === 'time_asc') {
    list.sort((a, b) => (a.start_time || 0) - (b.start_time || 0));
  } else if (state.sortMode === 'time_desc') {
    list.sort((a, b) => (b.start_time || 0) - (a.start_time || 0));
  } else if (state.sortMode === 'duration') {
    list.sort((a, b) => (b.duration || 0) - (a.duration || 0));
  }

  state.filteredAnomalies = list;
  renderAnomalyList();
}

function renderAnomalyList() {
  const container = document.getElementById('anomaly-items');
  if (!container) return;
  container.innerHTML = '';

  if (!state.filteredAnomalies || state.filteredAnomalies.length === 0) {
    container.innerHTML = '<div class="state-loader">暂无符合条件的争议样本</div>';
    return;
  }

  state.filteredAnomalies.forEach(item => {
    const div = document.createElement('div');
    div.className = `anomaly-card state-${(item.state || 'STATIC').toLowerCase()}`;
    div.id = `anomaly-item-${item.id}`;

    let reason = item.reason_desc || '边缘临界切片';
    let icon = '🔍';
    let badgeClass = 'chip-sta';

    if (item.manual_label) {
      reason = `已复核: ${item.manual_label}`;
      icon = '✓';
      badgeClass = 'chip-reviewed';
    } else if (item.anomaly_type === 'FP_SUSPECT' || (item.state === 'DYNAMIC' && item.avg_confidence === 0)) {
      reason = '疑似光影假阳性 (无目标置信度)';
      icon = '⚠️';
      badgeClass = 'chip-fp';
    } else if (item.anomaly_type === 'FN_SUSPECT' || (item.state === 'STATIC' && item.max_energy >= 1.5)) {
      reason = '疑似微动作漏判 (高能量静止)';
      icon = '🎯';
      badgeClass = 'chip-fn';
    } else if (item.duration < 3.0) {
      reason = '碎片跳变切片';
      icon = '⏱️';
      badgeClass = 'chip-jitter';
    }

    const fn = (item.filepath || '').split('\\').pop().split('/').pop();
    const durSec = (item.duration || 0).toFixed(1);
    const engVal = (item.max_energy || 0).toFixed(1);

    div.innerHTML = `
      <div class="anomaly-card-top">
        <span class="anomaly-id font-mono">#${item.id} • ${item.state}</span>
        <span class="anomaly-dur font-mono">${durSec}s</span>
      </div>
      <div class="anomaly-reason-tag ${badgeClass}">
        <span>${icon}</span>
        <span>${reason}</span>
      </div>
      <div class="anomaly-card-footer font-mono">
        <span>⚡ 峰值: ${engVal}</span>
        <span class="file-ellipsis" title="${item.filepath}">${fn}</span>
      </div>
    `;

    div.addEventListener('click', () => {
      document.querySelectorAll('.anomaly-card').forEach(el => el.classList.remove('selected'));
      div.classList.add('selected');
      loadFileSegments(item.file_id, item.filepath, item.id);
    });

    container.appendChild(div);
  });
}

// ================== 全量素材目录树 ==================
async function loadFileTree() {
  try {
    const res = await fetch('/api/tree');
    const tree = await res.json();
    const container = document.getElementById('tree-items');
    if (!container) return;
    container.innerHTML = '';

    if (!tree || tree.length === 0) {
      container.innerHTML = '<div class="state-loader">暂无素材文件数据</div>';
      return;
    }

    tree.forEach(dayGroup => {
      const dayDiv = document.createElement('div');
      dayDiv.className = 'tree-date-group';
      dayDiv.innerHTML = `<div class="tree-date-header">📅 日期: ${dayGroup.date}</div>`;

      dayGroup.cameras.forEach(cam => {
        const camDiv = document.createElement('div');
        camDiv.className = 'tree-cam-group';
        camDiv.innerHTML = `<div class="tree-cam-header">📹 机位 ${cam.cam_index} (${cam.files.length} 个视频)</div>`;

        cam.files.forEach(file => {
          const fDiv = document.createElement('div');
          fDiv.className = 'tree-file-item';
          fDiv.id = `tree-file-${file.id}`;
          fDiv.innerHTML = `
            <span class="tree-file-name font-mono">${file.filename}</span>
            <span class="tree-file-count">${file.reviewed_count}/${file.seg_count}</span>
          `;
          fDiv.addEventListener('click', () => {
            document.querySelectorAll('.tree-file-item').forEach(el => el.classList.remove('active'));
            fDiv.classList.add('active');
            loadFileSegments(file.id, file.filepath);
          });
          camDiv.appendChild(fDiv);
        });

        dayDiv.appendChild(camDiv);
      });

      container.appendChild(dayDiv);
    });
  } catch (err) {
    console.error('Failed to load file tree:', err);
  }
}

// ================== 加载单文件的所有切片与时间轴初始化 ==================
async function loadFileSegments(fileId, filepath, targetSegId = null) {
  try {
    const url = fileId ? `/api/file_segments?file_id=${fileId}` : `/api/file_segments?filepath=${encodeURIComponent(filepath)}`;
    const res = await fetch(url);
    const data = await res.json();

    if (!data.file) return;

    state.currentFile = data.file;
    state.currentSegments = data.segments || [];

    const fn = (state.currentFile.filepath || '').split('\\').pop().split('/').pop();
    const nameEl = document.getElementById('current-filename');
    if (nameEl) nameEl.innerText = fn;
    
    const chipEl = document.getElementById('current-status-chip');
    if (chipEl) {
      chipEl.innerText = `PRE: ${state.currentFile.prescreen_status} • ANA: ${state.currentFile.analysis_status}`;
    }

    // 解析视频基准墙钟时间与绝对偏移
    parseFileClockContext(fn);

    // 计算文件总时长与切片时间跨度
    const segs = state.currentSegments;
    const dur = state.currentFile.file_duration || (segs.length > 0 ? (segs[segs.length - 1].end_time - segs[0].start_time) : 300);
    state.timeline.fileDuration = Math.max(dur, 1);

    // 默认展示全景 Minimap，重置 Focus Track 可视窗口
    resetTimelineZoom();
    renderDualTimeline();

    // 定位目标切片
    let targetIndex = 0;
    if (targetSegId) {
      const found = state.currentSegments.findIndex(s => s.id === targetSegId);
      if (found >= 0) targetIndex = found;
    }
    selectSegment(targetIndex, true);
  } catch (err) {
    console.error('Failed to load file segments:', err);
  }
}

// 解析文件名中的真实墙钟时间 (例如 cam0_20260320180514.mp4 或 20260320_180514.mp4)
function parseFileClockContext(filename) {
  let baseHour = 0, baseMin = 0, baseSec = 0;
  const match = filename.match(/(\d{4})(\d{2})(\d{2})_?(\d{2})(\d{2})(\d{2})/);
  if (match) {
    const [_, y, m, d, hh, mm, ss] = match;
    baseHour = parseInt(hh, 10);
    baseMin = parseInt(mm, 10);
    baseSec = parseInt(ss, 10);
    state.timeline.fileOffsetSec = baseHour * 3600 + baseMin * 60 + baseSec;
    state.timeline.fileBaseClockDate = `${y}-${m}-${d}`;
  } else {
    state.timeline.fileOffsetSec = 0;
    state.timeline.fileBaseClockDate = '';
  }

  const dur = state.currentFile.file_duration || 300;
  const startClock = formatClockTime(state.timeline.fileOffsetSec);
  const endClock = formatClockTime(state.timeline.fileOffsetSec + dur);
  const clockEl = document.getElementById('current-file-clock');
  if (clockEl) {
    clockEl.innerText = `${startClock} ~ ${endClock}`;
  }
}

// 将日内绝对秒数转换为 hh:mm:ss 格式
function formatClockTime(absSec) {
  const totalSec = Math.floor(absSec) % 86400;
  const hh = Math.floor(totalSec / 3600);
  const mm = Math.floor((totalSec % 3600) / 60);
  const ss = Math.floor(totalSec % 60);
  return `${hh.toString().padStart(2, '0')}:${mm.toString().padStart(2, '0')}:${ss.toString().padStart(2, '0')}`;
}

// 格式化相对时间 mm:ss.s
function formatRelativeTime(sec) {
  const sClamped = Math.max(0, sec);
  const m = Math.floor(sClamped / 60);
  const s = (sClamped % 60).toFixed(1);
  return `${m.toString().padStart(2, '0')}:${s.padStart(4, '0')}`;
}

// ================== 双层联动时间轴渲染引擎 ==================

function renderDualTimeline() {
  renderMinimap();
  renderFocusTrack();
}

// 1. 上层 Minimap 轨
function renderMinimap() {
  const bar = document.getElementById('minimap-bar');
  if (!bar) return;
  bar.innerHTML = '';
  const segs = state.currentSegments;
  if (!segs || segs.length === 0) return;

  const totalDur = state.timeline.fileDuration;
  const durTxt = document.getElementById('minimap-duration-txt');
  if (durTxt) durTxt.innerText = formatRelativeTime(totalDur);

  const firstStart = segs[0].start_time;
  const isDayOffset = (firstStart >= 3600);
  const baseOffset = isDayOffset ? firstStart : 0;

  segs.forEach((s, idx) => {
    const block = document.createElement('div');
    block.className = 'minimap-seg';

    if (s.manual_label === 'FALSE_ALARM') {
      block.classList.add('state-fp');
    } else if (s.manual_label === 'MISSED_MOTION') {
      block.classList.add('state-fn');
    } else if (s.state === 'DYNAMIC') {
      block.classList.add('state-dynamic');
    } else {
      block.classList.add('state-static');
    }

    const localStart = isDayOffset ? (s.start_time - baseOffset) : s.start_time;
    const localDur = s.duration || (s.end_time - s.start_time);

    const leftPct = (localStart / totalDur) * 100;
    const widthPct = (localDur / totalDur) * 100;

    block.style.left = `${Math.max(0, Math.min(100, leftPct))}%`;
    block.style.width = `${Math.max(0.3, Math.min(100, widthPct))}%`;

    block.addEventListener('click', (e) => {
      e.stopPropagation();
      selectSegment(idx, true);
    });

    bar.appendChild(block);
  });

  updateMinimapLens();
}

function updateMinimapLens() {
  const lens = document.getElementById('minimap-lens');
  if (!lens) return;

  const totalDur = state.timeline.fileDuration;
  const vStart = state.timeline.viewStart;
  const vEnd = state.timeline.viewEnd;
  const vSpan = vEnd - vStart;

  const leftPct = (vStart / totalDur) * 100;
  const widthPct = (vSpan / totalDur) * 100;

  lens.style.left = `${Math.max(0, Math.min(100, leftPct))}%`;
  lens.style.width = `${Math.max(1.0, Math.min(100, widthPct))}%`;
}

// 2. 下层 Focus Track
function renderFocusTrack() {
  const segLayer = document.getElementById('focus-segments-layer');
  const ruler = document.getElementById('focus-ruler');
  if (!segLayer || !ruler) return;

  segLayer.innerHTML = '';
  ruler.innerHTML = '';

  const totalDur = state.timeline.fileDuration;
  const vStart = state.timeline.viewStart;
  const vEnd = state.timeline.viewEnd;
  const vSpan = Math.max(0.1, vEnd - vStart);

  const zoomFactor = (totalDur / vSpan).toFixed(1);
  const rangeTxt = document.getElementById('focus-range-txt');
  if (rangeTxt) {
    rangeTxt.innerText = `视窗范围: ${formatRelativeTime(vStart)} ~ ${formatRelativeTime(vEnd)} (放大 ${zoomFactor}x)`;
  }

  const numTicks = 6;
  for (let i = 0; i <= numTicks; i++) {
    const tSec = vStart + (vSpan * (i / numTicks));
    const leftPct = (i / numTicks) * 100;

    const tick = document.createElement('div');
    tick.className = 'ruler-tick';
    tick.style.left = `${leftPct}%`;

    const clockStr = formatClockTime(state.timeline.fileOffsetSec + tSec);
    const relStr = formatRelativeTime(tSec);

    tick.innerHTML = `
      <div class="ruler-line"></div>
      <span class="ruler-label font-mono">${clockStr} <small>(${relStr})</small></span>
    `;
    ruler.appendChild(tick);
  }

  const segs = state.currentSegments;
  if (!segs || segs.length === 0) return;

  const firstStart = segs[0].start_time;
  const isDayOffset = (firstStart >= 3600);
  const baseOffset = isDayOffset ? firstStart : 0;

  segs.forEach((s, idx) => {
    const localStart = isDayOffset ? (s.start_time - baseOffset) : s.start_time;
    const localDur = s.duration || (s.end_time - s.start_time);
    const localEnd = localStart + localDur;

    if (localEnd < vStart || localStart > vEnd) return;

    const block = document.createElement('div');
    block.className = 'focus-seg-block';
    block.id = `focus-seg-${idx}`;

    if (s.manual_label === 'FALSE_ALARM') {
      block.classList.add('state-fp');
    } else if (s.manual_label === 'MISSED_MOTION') {
      block.classList.add('state-fn');
    } else if (s.state === 'DYNAMIC') {
      block.classList.add('state-dynamic');
    } else {
      block.classList.add('state-static');
    }

    if (idx === state.selectedSegIndex) {
      block.classList.add('selected');
    }

    const segLeftInView = Math.max(0, localStart - vStart);
    const segRightInView = Math.min(vSpan, localEnd - vStart);
    const leftPct = (segLeftInView / vSpan) * 100;
    const widthPct = ((segRightInView - segLeftInView) / vSpan) * 100;

    block.style.left = `${leftPct}%`;
    block.style.width = `${Math.max(0.6, widthPct)}%`;

    if (widthPct > 5.0) {
      const labelSpan = document.createElement('span');
      labelSpan.className = 'focus-seg-label font-mono';
      labelSpan.innerText = `#${s.id} ${s.state} (${localDur.toFixed(1)}s)`;
      block.appendChild(labelSpan);
    }

    block.addEventListener('click', (e) => {
      e.stopPropagation();
      selectSegment(idx, false);
    });

    block.addEventListener('mouseenter', () => {
      const hoverEl = document.getElementById('timeline-hover-info');
      if (hoverEl) {
        hoverEl.innerText = `切片 #${s.id} | ${formatRelativeTime(localStart)} ~ ${formatRelativeTime(localEnd)} (${localDur.toFixed(1)}s) | 状态: ${s.state} | 能量: ${(s.max_energy || 0).toFixed(1)}`;
      }
    });

    segLayer.appendChild(block);
  });
}

function focusCurrentSegment() {
  if (state.selectedSegIndex < 0 || state.selectedSegIndex >= state.currentSegments.length) return;
  const seg = state.currentSegments[state.selectedSegIndex];

  const firstStart = state.currentSegments[0].start_time;
  const isDayOffset = (firstStart >= 3600);
  const baseOffset = isDayOffset ? firstStart : 0;

  const localStart = isDayOffset ? (seg.start_time - baseOffset) : seg.start_time;
  const localDur = seg.duration || (seg.end_time - seg.start_time);
  const localMid = localStart + localDur / 2.0;

  const targetSpan = Math.min(state.timeline.fileDuration, Math.max(16.0, localDur * 4.5));
  
  let newStart = localMid - targetSpan / 2.0;
  let newEnd = localMid + targetSpan / 2.0;

  if (newStart < 0) {
    newStart = 0;
    newEnd = Math.min(state.timeline.fileDuration, targetSpan);
  } else if (newEnd > state.timeline.fileDuration) {
    newEnd = state.timeline.fileDuration;
    newStart = Math.max(0, newEnd - targetSpan);
  }

  state.timeline.viewStart = newStart;
  state.timeline.viewEnd = newEnd;

  updateMinimapLens();
  renderFocusTrack();
}

function zoomTimeline(factor, anchorRatio = 0.5) {
  const totalDur = state.timeline.fileDuration;
  const curSpan = state.timeline.viewEnd - state.timeline.viewStart;
  let newSpan = curSpan * factor;

  newSpan = Math.max(4.0, Math.min(totalDur, newSpan));

  const anchorSec = state.timeline.viewStart + curSpan * anchorRatio;
  let newStart = anchorSec - newSpan * anchorRatio;
  let newEnd = anchorSec + newSpan * (1.0 - anchorRatio);

  if (newStart < 0) {
    newStart = 0;
    newEnd = Math.min(totalDur, newSpan);
  } else if (newEnd > totalDur) {
    newEnd = totalDur;
    newStart = Math.max(0, newEnd - newSpan);
  }

  state.timeline.viewStart = newStart;
  state.timeline.viewEnd = newEnd;

  updateMinimapLens();
  renderFocusTrack();
}

function resetTimelineZoom() {
  state.timeline.viewStart = 0;
  state.timeline.viewEnd = state.timeline.fileDuration;
  updateMinimapLens();
  renderFocusTrack();
}

function initMinimapInteractions() {
  const wrap = document.getElementById('minimap-track-wrap');
  const lens = document.getElementById('minimap-lens');
  if (!wrap || !lens) return;

  wrap.addEventListener('click', (e) => {
    if (state.timeline.isDraggingLens) return;
    const rect = wrap.getBoundingClientRect();
    const clickRatio = (e.clientX - rect.left) / rect.width;
    const clickSec = clickRatio * state.timeline.fileDuration;

    const span = state.timeline.viewEnd - state.timeline.viewStart;
    let newStart = clickSec - span / 2.0;
    let newEnd = clickSec + span / 2.0;

    if (newStart < 0) {
      newStart = 0;
      newEnd = Math.min(state.timeline.fileDuration, span);
    } else if (newEnd > state.timeline.fileDuration) {
      newEnd = state.timeline.fileDuration;
      newStart = Math.max(0, newEnd - span);
    }

    state.timeline.viewStart = newStart;
    state.timeline.viewEnd = newEnd;
    updateMinimapLens();
    renderFocusTrack();
  });

  lens.addEventListener('mousedown', (e) => {
    e.stopPropagation();
    state.timeline.isDraggingLens = true;
    state.timeline.dragStartX = e.clientX;
    state.timeline.lensDragStartViewStart = state.timeline.viewStart;
    document.body.style.userSelect = 'none';
  });

  window.addEventListener('mousemove', (e) => {
    if (!state.timeline.isDraggingLens) return;
    const wrapRect = wrap.getBoundingClientRect();
    const deltaX = e.clientX - state.timeline.dragStartX;
    const deltaSec = (deltaX / wrapRect.width) * state.timeline.fileDuration;

    const span = state.timeline.viewEnd - state.timeline.viewStart;
    let newStart = state.timeline.lensDragStartViewStart + deltaSec;
    let newEnd = newStart + span;

    if (newStart < 0) {
      newStart = 0;
      newEnd = span;
    } else if (newEnd > state.timeline.fileDuration) {
      newEnd = state.timeline.fileDuration;
      newStart = newEnd - span;
    }

    state.timeline.viewStart = newStart;
    state.timeline.viewEnd = newEnd;
    updateMinimapLens();
    renderFocusTrack();
  });

  window.addEventListener('mouseup', () => {
    if (state.timeline.isDraggingLens) {
      state.timeline.isDraggingLens = false;
      document.body.style.userSelect = '';
    }
  });
}

function initFocusTrackInteractions() {
  const trackWrapper = document.getElementById('focus-track-wrapper');
  if (!trackWrapper) return;

  trackWrapper.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = trackWrapper.getBoundingClientRect();
    const mouseRatio = (e.clientX - rect.left) / rect.width;
    const factor = e.deltaY > 0 ? 1.25 : 0.8;
    zoomTimeline(factor, mouseRatio);
  }, { passive: false });

  trackWrapper.addEventListener('mousedown', (e) => {
    if (e.target.closest('.focus-seg-block')) return;
    state.timeline.isPanningFocus = true;
    state.timeline.panStartX = e.clientX;
    state.timeline.panStartViewStart = state.timeline.viewStart;
    trackWrapper.style.cursor = 'grabbing';
  });

  window.addEventListener('mousemove', (e) => {
    if (!state.timeline.isPanningFocus) return;
    const rect = trackWrapper.getBoundingClientRect();
    const deltaX = e.clientX - state.timeline.panStartX;
    const span = state.timeline.viewEnd - state.timeline.viewStart;
    const deltaSec = -(deltaX / rect.width) * span;

    let newStart = state.timeline.panStartViewStart + deltaSec;
    let newEnd = newStart + span;

    if (newStart < 0) {
      newStart = 0;
      newEnd = span;
    } else if (newEnd > state.timeline.fileDuration) {
      newEnd = state.timeline.fileDuration;
      newStart = newEnd - span;
    }

    state.timeline.viewStart = newStart;
    state.timeline.viewEnd = newEnd;
    updateMinimapLens();
    renderFocusTrack();
  });

  window.addEventListener('mouseup', () => {
    if (state.timeline.isPanningFocus) {
      state.timeline.isPanningFocus = false;
      if (trackWrapper) trackWrapper.style.cursor = '';
    }
  });
}

// ================== 切片选中与三联画廊安全防裂加载 ==================

function selectSegment(index, autoFocus = false) {
  if (index < 0 || index >= state.currentSegments.length) return;
  state.selectedSegIndex = index;
  const seg = state.currentSegments[index];

  document.querySelectorAll('.focus-seg-block').forEach(b => b.classList.remove('selected'));
  const targetBlock = document.getElementById(`focus-seg-${index}`);
  if (targetBlock) targetBlock.classList.add('selected');

  if (autoFocus) {
    focusCurrentSegment();
  }

  const tagsWrap = document.getElementById('current-segment-tags');
  if (tagsWrap) {
    tagsWrap.innerHTML = '';

    const stateChip = document.createElement('span');
    stateChip.className = `chip ${seg.state === 'DYNAMIC' ? 'chip-dyn' : 'chip-sta'}`;
    stateChip.innerText = `● 原判: ${seg.state}`;
    tagsWrap.appendChild(stateChip);

    const durChip = document.createElement('span');
    durChip.className = 'chip font-mono';
    durChip.innerText = `⏱️ ${(seg.duration || 0).toFixed(1)}s`;
    tagsWrap.appendChild(durChip);

    const energyChip = document.createElement('span');
    energyChip.className = 'chip font-mono';
    energyChip.innerText = `⚡ 能量: ${(seg.max_energy || 0).toFixed(2)}`;
    tagsWrap.appendChild(energyChip);

    if (seg.manual_label) {
      const revChip = document.createElement('span');
      revChip.className = 'chip chip-reviewed';
      revChip.innerText = `✓ 已复核: ${seg.manual_label}`;
      tagsWrap.appendChild(revChip);
    }
  }

  const yoloTags = document.getElementById('yolo-tags');
  if (yoloTags) {
    yoloTags.innerHTML = '<span class="inspector-placeholder">点击「⚡ 运行 YOLO 强化画框 (快捷键 Y)」即刻进行多目标分析与边界框定位</span>';
  }

  loadFramesForSegment(seg);
}

function loadFramesForSegment(seg) {
  const fp = state.currentFile.filepath;
  const tStart = seg.start_time;
  const tMid = (seg.start_time + seg.end_time) / 2.0;
  const tEnd = Math.max(seg.end_time - 0.1, seg.start_time);

  const firstStart = state.currentSegments[0].start_time;
  const isDayOffset = (firstStart >= 3600);
  const baseOffset = isDayOffset ? firstStart : 0;

  const relStart = isDayOffset ? (tStart - baseOffset) : tStart;
  const relMid = isDayOffset ? (tMid - baseOffset) : tMid;
  const relEnd = isDayOffset ? (tEnd - baseOffset) : tEnd;

  setElementText('ts-start', `${relStart.toFixed(2)}s`);
  setElementText('ts-mid', `${relMid.toFixed(2)}s`);
  setElementText('ts-end', `${relEnd.toFixed(2)}s`);

  const clockStart = formatClockTime(state.timeline.fileOffsetSec + relStart);
  const clockMid = formatClockTime(state.timeline.fileOffsetSec + relMid);
  const clockEnd = formatClockTime(state.timeline.fileOffsetSec + relEnd);

  setElementText('clock-start', clockStart);
  setElementText('clock-mid', clockMid);
  setElementText('clock-end', clockEnd);

  setSafeFrameImage('start', fp, tStart);
  setSafeFrameImage('mid', fp, tMid);
  setSafeFrameImage('end', fp, tEnd);
}

function setElementText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}

function setSafeFrameImage(pos, filepath, timestamp) {
  const img = document.getElementById(`img-${pos}`);
  const shimmer = document.getElementById(`shimmer-${pos}`);
  const placeholder = document.getElementById(`placeholder-${pos}`);

  if (!img || !placeholder) return;

  img.classList.add('hidden');
  placeholder.classList.remove('hidden');
  placeholder.innerHTML = `
    <span class="ph-icon">⏳</span>
    <span class="ph-text">正在抽取高清关键帧...</span>
  `;
  if (shimmer) shimmer.classList.remove('hidden');

  const src = `/api/frame?filepath=${encodeURIComponent(filepath)}&t=${timestamp.toFixed(3)}&w=640`;

  const tempImg = new Image();
  tempImg.onload = () => {
    img.src = src;
    img.classList.remove('hidden');
    placeholder.classList.add('hidden');
    if (shimmer) shimmer.classList.add('hidden');
  };
  tempImg.onerror = () => {
    img.classList.add('hidden');
    if (shimmer) shimmer.classList.add('hidden');
    placeholder.classList.remove('hidden');
    placeholder.innerHTML = `
      <span class="ph-icon" style="color: #f43f5e;">⚠️</span>
      <span class="ph-text" style="color: #fca5a5;">关键帧抽帧离线</span>
      <button class="btn btn-glass btn-sm" style="margin-top: 6px; font-size: 11px;" onclick="setSafeFrameImage('${pos}', '${filepath.replace(/\\/g, '\\\\')}', ${timestamp})">↻ 点击重试</button>
    `;
  };
  tempImg.src = src;
}

// ================== YOLO 强化检测与动图预览 ==================

async function runYoloEnhance() {
  if (state.selectedSegIndex < 0) return;
  const seg = state.currentSegments[state.selectedSegIndex];
  const tMid = (seg.start_time + seg.end_time) / 2.0;

  const btn = document.getElementById('btn-yolo-enhance');
  if (!btn) return;
  const oldText = btn.innerHTML;
  btn.innerHTML = '<span class="spin-ring" style="width: 14px; height: 14px; display: inline-block;"></span> 推理中...';
  btn.disabled = true;

  try {
    const res = await fetch('/api/yolo_detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filepath: state.currentFile.filepath,
        timestamp: tMid
      })
    });
    const data = await res.json();
    if (data.image_url) {
      const imgMid = document.getElementById('img-mid');
      const placeholderMid = document.getElementById('placeholder-mid');
      if (imgMid) {
        imgMid.src = data.image_url;
        imgMid.classList.remove('hidden');
      }
      if (placeholderMid) placeholderMid.classList.add('hidden');

      const tagsWrap = document.getElementById('yolo-tags');
      if (tagsWrap) {
        tagsWrap.innerHTML = '';
        if (!data.objects || data.objects.length === 0) {
          tagsWrap.innerHTML = '<span class="inspector-placeholder">⚠️ 未检出明确目标（高度疑似为光影/晃动假动作）</span>';
          showToast('检测完成：未检出主体目标', 'info');
        } else {
          data.objects.forEach(obj => {
            const span = document.createElement('span');
            span.className = 'obj-tag';
            const icon = obj.class === 'person' ? '👤' : (['cat', 'dog'].includes(obj.class) ? '🐾' : '🎯');
            span.innerHTML = `${icon} <strong>${obj.class}</strong> ${(obj.confidence * 100).toFixed(0)}%`;
            tagsWrap.appendChild(span);
          });
          showToast(`检测完成：命中 ${data.objects.length} 个目标`, 'success');
        }
      }
    }
  } catch (err) {
    console.error('YOLO enhance failed:', err);
    showToast('YOLO 增强检测失败', 'error');
  } finally {
    btn.innerHTML = oldText;
    btn.disabled = false;
  }
}

function showClipPreview() {
  if (state.selectedSegIndex < 0) return;
  const seg = state.currentSegments[state.selectedSegIndex];
  const modal = document.getElementById('preview-modal');
  const img = document.getElementById('modal-preview-img');
  const loading = document.getElementById('modal-loading');

  if (!modal || !img || !loading) return;

  modal.classList.remove('hidden');
  img.src = '';
  loading.style.display = 'flex';

  const clipUrl = `/api/clip?filepath=${encodeURIComponent(state.currentFile.filepath)}&start=${seg.start_time}&end=${seg.end_time}`;
  const tempImg = new Image();
  tempImg.onload = () => {
    img.src = clipUrl;
    loading.style.display = 'none';
  };
  tempImg.onerror = () => {
    loading.innerHTML = '<span style="color: #f43f5e;">⚠️ WebP 动图压制失败或素材离线</span>';
  };
  tempImg.src = clipUrl;
}

// ================== 裁决打标与自动步进 ==================

async function submitCurrentReview(manualLabel) {
  if (state.selectedSegIndex < 0) return;
  const seg = state.currentSegments[state.selectedSegIndex];

  try {
    const res = await fetch('/api/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        segment_id: seg.id,
        manual_label: manualLabel,
        notes: ''
      })
    });
    const data = await res.json();
    if (data.success) {
      seg.manual_label = manualLabel;
      renderDualTimeline();
      loadOverview();
      loadCategoryBadges();
      loadArchiveStats();

      const labelMap = {
        'CONFIRMED_MOTION': '动态确认 (TP)',
        'FALSE_ALARM': '误报光影 (FP)',
        'MISSED_MOTION': '漏检补录 (FN)',
        'CONFIRMED_STATIC': '纯静确认 (TN)'
      };
      const labelName = labelMap[manualLabel] || manualLabel;
      showToast(`已标记为 [${labelName}] 并自动归档真实代表帧`, 'success');

      selectNextSegment(1);
    }
  } catch (err) {
    console.error('Submit review failed:', err);
    showToast('提交打标失败', 'error');
  }
}

function selectNextSegment(step) {
  const nextIdx = state.selectedSegIndex + step;
  if (nextIdx >= 0 && nextIdx < state.currentSegments.length) {
    selectSegment(nextIdx, true);
  }
}

// ================== 重渲染状态机与交互 ==================
let rerenderPollTimer = null;
let currentRerenderOutputFile = '';

function getActiveDateAndCam() {
  if (state.currentFile) {
    return { date: state.currentFile.date, cam: state.currentFile.cam_index };
  }
  if (state.anomalies && state.anomalies.length > 0) {
    return { date: state.anomalies[0].date, cam: state.anomalies[0].cam_index };
  }
  return { date: '20260320', cam: 0 };
}

function openRerenderModal() {
  const { date, cam } = getActiveDateAndCam();
  const descEl = document.getElementById('rerender-target-desc');
  if (descEl) descEl.innerText = `${date} (cam${cam})`;

  const statusEl = document.getElementById('rerender-file-check-status');
  if (statusEl) {
    statusEl.className = 'status-chip';
    statusEl.innerText = '就绪待命';
  }

  setElementText('rerender-stage-text', '准备重新浓缩');
  setElementText('rerender-pct-text', '0%');
  setElementText('rerender-batch-text', '批次: 0 / 0');
  setElementText('rerender-time-text', '已耗时: 0.0s');

  const fill = document.getElementById('rerender-progress-fill');
  if (fill) {
    fill.style.width = '0%';
    fill.style.background = 'linear-gradient(90deg, #06b6d4, #3b82f6)';
  }

  const alertEl = document.getElementById('rerender-missing-alert');
  if (alertEl) alertEl.classList.add('hidden');
  const listEl = document.getElementById('rerender-missing-list');
  if (listEl) listEl.innerHTML = '';

  const btnStart = document.getElementById('btn-rerender-start');
  if (btnStart) {
    btnStart.classList.remove('hidden');
    btnStart.disabled = false;
  }
  const btnOpenDir = document.getElementById('btn-rerender-open-dir');
  if (btnOpenDir) btnOpenDir.classList.add('hidden');
  const btnCancel = document.getElementById('btn-rerender-cancel');
  if (btnCancel) btnCancel.innerText = '关闭';

  const modal = document.getElementById('rerender-modal');
  if (modal) modal.classList.remove('hidden');

  checkCurrentRerenderStatus(date, cam);
}

function closeRerenderModal() {
  if (rerenderPollTimer) {
    clearInterval(rerenderPollTimer);
    rerenderPollTimer = null;
  }
  const modal = document.getElementById('rerender-modal');
  if (modal) modal.classList.add('hidden');
}

function cancelOrCloseRerender() {
  const { date, cam } = getActiveDateAndCam();
  const btn = document.getElementById('btn-rerender-cancel');
  if (btn && btn.innerText === '中断任务') {
    fetch('/api/rerender_cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ date, cam_index: cam })
    }).then(() => {
      showToast('已请求中断重渲染任务', 'info');
      btn.innerText = '关闭';
    });
  } else {
    closeRerenderModal();
  }
}

async function checkCurrentRerenderStatus(date, cam) {
  try {
    const res = await fetch(`/api/rerender_status?date=${date}&cam_index=${cam}`);
    const data = await res.json();
    if (['CHECKING', 'RENDERING', 'CONCATING'].includes(data.status)) {
      applyRerenderStatus(data);
      startPollingRerender(date, cam);
    }
  } catch (err) {
    console.error('Check rerender status error:', err);
  }
}

async function startRerender() {
  const { date, cam } = getActiveDateAndCam();
  const btnStart = document.getElementById('btn-rerender-start');
  if (btnStart) btnStart.disabled = true;
  const btnCancel = document.getElementById('btn-rerender-cancel');
  if (btnCancel) btnCancel.innerText = '中断任务';

  try {
    const res = await fetch('/api/rerender', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        date: date,
        cam_index: cam,
        version: 'reviewed'
      })
    });
    const data = await res.json();
    if (data.error && data.error !== 'Task already running') {
      showToast(`启动失败: ${data.error}`, 'error');
      if (btnStart) btnStart.disabled = false;
      if (btnCancel) btnCancel.innerText = '关闭';
      return;
    }
    showToast(`已提交 ${date} cam${cam} 重浓缩任务`, 'info');
    startPollingRerender(date, cam);
  } catch (err) {
    console.error('Start rerender failed:', err);
    showToast('提交重浓缩请求异常', 'error');
    if (btnStart) btnStart.disabled = false;
  }
}

function startPollingRerender(date, cam) {
  if (rerenderPollTimer) clearInterval(rerenderPollTimer);
  rerenderPollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/rerender_status?date=${date}&cam_index=${cam}`);
      const st = await res.json();
      applyRerenderStatus(st);

      if (['COMPLETED', 'FAILED', 'CANCELLED', 'IDLE'].includes(st.status)) {
        clearInterval(rerenderPollTimer);
        rerenderPollTimer = null;
        const btnCancel = document.getElementById('btn-rerender-cancel');
        if (btnCancel) btnCancel.innerText = '关闭';
        const btnStart = document.getElementById('btn-rerender-start');
        if (btnStart) btnStart.disabled = false;
      }
    } catch (err) {
      console.error('Poll rerender status failed:', err);
    }
  }, 900);
}

function applyRerenderStatus(st) {
  const checkChip = document.getElementById('rerender-file-check-status');
  const stageText = document.getElementById('rerender-stage-text');
  const pctText = document.getElementById('rerender-pct-text');
  const barFill = document.getElementById('rerender-progress-fill');
  const batchText = document.getElementById('rerender-batch-text');
  const timeText = document.getElementById('rerender-time-text');
  const missingAlert = document.getElementById('rerender-missing-alert');
  const missingList = document.getElementById('rerender-missing-list');
  const btnStart = document.getElementById('btn-rerender-start');
  const btnOpenDir = document.getElementById('btn-rerender-open-dir');

  if (timeText) timeText.innerText = `已耗时: ${(st.elapsed_s || 0).toFixed(1)}s`;
  if (pctText) pctText.innerText = `${st.progress || 0}%`;
  if (barFill) barFill.style.width = `${st.progress || 0}%`;

  if (st.missing_files && st.missing_files.length > 0) {
    if (missingAlert) missingAlert.classList.remove('hidden');
    if (missingList) {
      missingList.innerHTML = st.missing_files.map(f => `<div>• ${f.split('\\').pop().split('/').pop()}</div>`).join('');
    }
    if (checkChip) {
      checkChip.className = 'status-chip';
      checkChip.style.background = 'rgba(245, 158, 11, 0.2)';
      checkChip.style.color = '#f59e0b';
      checkChip.innerText = `⚠️ 缺失 ${st.missing_files.length} 个物理文件 (已自愈跳过)`;
    }
  } else if (st.status !== 'IDLE' && checkChip) {
    checkChip.className = 'status-chip';
    checkChip.style.background = 'rgba(16, 185, 129, 0.2)';
    checkChip.style.color = '#34d399';
    checkChip.innerText = '✅ 全部源文件物理校验通过';
  }

  if (st.status === 'CHECKING' && stageText) {
    stageText.innerText = '正在探测素材物理存在性与可读性...';
  } else if (st.status === 'RENDERING') {
    if (stageText) stageText.innerText = `正在执行硬件加速切片压制 (批次 ${st.current_batch}/${st.total_batches})...`;
    if (batchText) batchText.innerText = `批次: ${st.current_batch} / ${st.total_batches}`;
  } else if (st.status === 'CONCATING' && stageText) {
    stageText.innerText = '正在合并最终成片并封装元数据...';
  } else if (st.status === 'COMPLETED') {
    if (stageText) stageText.innerText = `🎉 浓缩压制完成！成片体积: ${st.file_size_mb || 0} MB`;
    if (barFill) barFill.style.background = '#10b981';
    if (btnStart) btnStart.classList.add('hidden');
    if (btnOpenDir) btnOpenDir.classList.remove('hidden');
    currentRerenderOutputFile = st.output_file;
    showToast('Vlog 成片重新浓缩成功！', 'success');
  } else if (st.status === 'FAILED') {
    if (stageText) stageText.innerText = `❌ 重渲染失败: ${st.error || '未知错误'}`;
    if (barFill) barFill.style.background = '#f43f5e';
    showToast(`重渲染失败: ${st.error}`, 'error');
  } else if (st.status === 'CANCELLED' && stageText) {
    stageText.innerText = '已中止任务';
  }
}

function openRerenderOutputDir() {
  if (!currentRerenderOutputFile) return;
  fetch(`/api/open_output_dir?filepath=${encodeURIComponent(currentRerenderOutputFile)}`)
    .then(r => r.json())
    .then(data => {
      if (data.success) {
        showToast('已在文件管理器中定位成片', 'info');
      } else {
        showToast('打开文件路径失败', 'error');
      }
    });
}

// ================== 真实反馈帧归档管理 ==================

async function loadArchiveStats() {
  try {
    const res = await fetch('/api/archive_stats');
    const data = await res.json();
    const countBadge = document.getElementById('archive-count-badge');
    if (countBadge) countBadge.innerText = data.total_images || 0;

    const dirEl = document.getElementById('archive-dir-path');
    if (dirEl) dirEl.innerText = data.archive_dir || 'data/feedback_archive';

    const totalImgEl = document.getElementById('archive-total-images');
    if (totalImgEl) totalImgEl.innerText = `${data.total_images || 0} 张`;

    const totalSizeEl = document.getElementById('archive-total-size');
    if (totalSizeEl) totalSizeEl.innerText = `${data.total_size_mb || 0} MB`;

    const breakdownEl = document.getElementById('archive-label-breakdown');
    if (breakdownEl) {
      breakdownEl.innerHTML = '';
      const counts = data.label_counts || {};
      if (Object.keys(counts).length === 0) {
        breakdownEl.innerHTML = '<span style="color: var(--text-muted); font-size: 11px;">暂无打标归档数据</span>';
      } else {
        const labelStyles = {
          'CONFIRMED_MOTION': { text: '动态确认', bg: 'rgba(16, 185, 129, 0.2)', color: '#34d399' },
          'VERIFIED_MOTION': { text: '动态确认', bg: 'rgba(16, 185, 129, 0.2)', color: '#34d399' },
          'FALSE_ALARM': { text: '光影误报', bg: 'rgba(244, 63, 94, 0.2)', color: '#f87171' },
          'MISSED_MOTION': { text: '漏报补录', bg: 'rgba(59, 130, 246, 0.2)', color: '#60a5fa' },
          'CONFIRMED_STATIC': { text: '纯静确认', bg: 'rgba(148, 163, 184, 0.2)', color: '#94a3b8' },
        };
        for (const [lbl, cnt] of Object.entries(counts)) {
          const style = labelStyles[lbl] || { text: lbl, bg: 'rgba(255,255,255,0.1)', color: '#fff' };
          const span = document.createElement('span');
          span.style.cssText = `padding: 2px 8px; border-radius: 4px; font-size: 11px; background: ${style.bg}; color: ${style.color};`;
          span.innerText = `${style.text}: ${cnt}`;
          breakdownEl.appendChild(span);
        }
      }
    }
  } catch (err) {
    console.error('Failed to load archive stats:', err);
  }
}

function openArchiveModal() {
  loadArchiveStats();
  const modal = document.getElementById('archive-modal');
  if (modal) modal.classList.remove('hidden');
}

function closeArchiveModal() {
  const modal = document.getElementById('archive-modal');
  if (modal) modal.classList.add('hidden');
}

async function triggerBatchArchive() {
  const btn = document.getElementById('btn-archive-batch-run');
  if (!btn) return;
  const oldText = btn.innerHTML;
  btn.innerHTML = '<span class="spin-ring" style="width: 14px; height: 14px; display: inline-block;"></span> 归档中...';
  btn.disabled = true;

  try {
    const res = await fetch('/api/archive_all', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force: false })
    });
    const data = await res.json();
    showToast(`补齐归档完成：新增 ${data.archived} 张，跳过/失败 ${data.failed} 张`, 'success');
    loadArchiveStats();
  } catch (err) {
    console.error('Batch archive failed:', err);
    showToast('批量归档失败', 'error');
  } finally {
    btn.innerHTML = oldText;
    btn.disabled = false;
  }
}

// ================== 🚀 高清关键帧细节放大查看器 (Lightbox Engine) ==================

const lightboxState = {
  isOpen: false,
  scale: 1.0,
  minScale: 0.5,
  maxScale: 6.0,
  translateX: 0,
  translateY: 0,
  isPanning: false,
  panStartX: 0,
  panStartY: 0,
  startTranslateX: 0,
  startTranslateY: 0,
  currentPos: '',
  currentTimestamp: 0
};

function initLightbox() {
  const modal = document.getElementById('lightbox-modal');
  const btnClose = document.getElementById('btn-lightbox-close');
  const btnZoomIn = document.getElementById('btn-lightbox-zoom-in');
  const btnZoomOut = document.getElementById('btn-lightbox-zoom-out');
  const btnReset = document.getElementById('btn-lightbox-reset');
  const btnActual = document.getElementById('btn-lightbox-actual');
  const viewport = document.getElementById('lightbox-viewport');
  const canvas = document.getElementById('lightbox-canvas');

  if (btnClose) btnClose.addEventListener('click', closeLightbox);
  if (btnZoomIn) btnZoomIn.addEventListener('click', () => zoomLightbox(1.25));
  if (btnZoomOut) btnZoomOut.addEventListener('click', () => zoomLightbox(0.8));
  if (btnReset) btnReset.addEventListener('click', resetLightboxZoom);
  if (btnActual) btnActual.addEventListener('click', setLightboxActualSize);

  // 点击背景外侧关闭
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeLightbox();
    });
  }

  // 绑定三联卡片头部放大按钮与卡片容器点击
  ['start', 'mid', 'end'].forEach(pos => {
    const btnZoom = document.getElementById(`btn-zoom-${pos}`);
    if (btnZoom) {
      btnZoom.addEventListener('click', (e) => {
        e.stopPropagation();
        openLightbox(pos);
      });
    }

    const wrap = document.getElementById(`wrap-${pos}`);
    if (wrap) {
      wrap.addEventListener('click', (e) => {
        const img = document.getElementById(`img-${pos}`);
        if (img && !img.classList.contains('hidden') && img.src) {
          openLightbox(pos);
        }
      });
    }
  });

  // 滚轮缩放画布 (以视窗中心或指针为锚点)
  if (viewport) {
    viewport.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const mouseX = e.clientX - rect.left - rect.width / 2;
      const mouseY = e.clientY - rect.top - rect.height / 2;

      const factor = e.deltaY < 0 ? 1.2 : 0.833;
      zoomLightboxAt(factor, mouseX, mouseY);
    }, { passive: false });
  }

  // 鼠标拖拽平移
  if (canvas) {
    canvas.addEventListener('mousedown', (e) => {
      e.preventDefault();
      lightboxState.isPanning = true;
      lightboxState.panStartX = e.clientX;
      lightboxState.panStartY = e.clientY;
      lightboxState.startTranslateX = lightboxState.translateX;
      lightboxState.startTranslateY = lightboxState.translateY;
      canvas.classList.add('panning');
    });

    window.addEventListener('mousemove', (e) => {
      if (!lightboxState.isPanning) return;
      const dx = e.clientX - lightboxState.panStartX;
      const dy = e.clientY - lightboxState.panStartY;
      lightboxState.translateX = lightboxState.startTranslateX + dx;
      lightboxState.translateY = lightboxState.startTranslateY + dy;
      applyLightboxTransform();
    });

    window.addEventListener('mouseup', () => {
      if (lightboxState.isPanning) {
        lightboxState.isPanning = false;
        if (canvas) canvas.classList.remove('panning');
      }
    });

    // 双击居中复位或 2.0x 放大切换
    canvas.addEventListener('dblclick', (e) => {
      e.preventDefault();
      if (Math.abs(lightboxState.scale - 1.0) < 0.15) {
        zoomLightbox(2.0);
      } else {
        resetLightboxZoom();
      }
    });
  }
}

function openLightbox(pos) {
  if (state.selectedSegIndex < 0 || !state.currentFile) return;
  const seg = state.currentSegments[state.selectedSegIndex];
  const fp = state.currentFile.filepath;
  const modal = document.getElementById('lightbox-modal');
  const imgEl = document.getElementById('lightbox-img');
  const spinner = document.getElementById('lightbox-loading');
  const titleEl = document.getElementById('lightbox-title');
  const metaEl = document.getElementById('lightbox-meta');

  if (!modal || !imgEl) return;

  lightboxState.isOpen = true;
  lightboxState.currentPos = pos;
  resetLightboxZoom();

  // 计算时间戳与语义标题
  let timestamp = seg.start_time;
  let posName = '起点进入时刻 (Start Frame)';
  if (pos === 'mid') {
    timestamp = (seg.start_time + seg.end_time) / 2.0;
    posName = '★ 核心峰值能量时刻 (Peak Frame / YOLO 检验点)';
  } else if (pos === 'end') {
    timestamp = Math.max(seg.end_time - 0.1, seg.start_time);
    posName = '终点离开时刻 (End Frame)';
  }
  lightboxState.currentTimestamp = timestamp;

  const firstStart = state.currentSegments[0].start_time;
  const isDayOffset = (firstStart >= 3600);
  const baseOffset = isDayOffset ? firstStart : 0;
  const relTime = isDayOffset ? (timestamp - baseOffset) : timestamp;
  const clockTime = formatClockTime(state.timeline.fileOffsetSec + relTime);
  const fn = (fp || '').split('\\').pop().split('/').pop();

  if (titleEl) titleEl.innerText = posName;
  if (metaEl) metaEl.innerText = `${clockTime} (${relTime.toFixed(2)}s) • ${fn}`;

  modal.classList.remove('hidden');

  // 先挂载当前已有图片作为瞬时底图预览，防止白屏等待
  const cardImg = document.getElementById(`img-${pos}`);
  if (cardImg && cardImg.src && !cardImg.classList.contains('hidden')) {
    imgEl.src = cardImg.src;
  }

  // 若中继帧已具备 YOLO 增强画框结果，直接采用画框图供用户微距检阅
  if (pos === 'mid' && cardImg && cardImg.src && (cardImg.src.includes('yolo') || cardImg.src.includes('data:image'))) {
    imgEl.src = cardImg.src;
    if (spinner) spinner.classList.add('hidden');
    return;
  }

  // 异步获取 1080P/超高清原画关键帧
  if (spinner) spinner.classList.remove('hidden');
  const hdSrc = `/api/frame?filepath=${encodeURIComponent(fp)}&t=${timestamp.toFixed(3)}&w=1920`;
  const tempImg = new Image();
  tempImg.onload = () => {
    imgEl.src = hdSrc;
    if (spinner) spinner.classList.add('hidden');
  };
  tempImg.onerror = () => {
    if (spinner) spinner.classList.add('hidden');
  };
  tempImg.src = hdSrc;
}

function closeLightbox() {
  lightboxState.isOpen = false;
  const modal = document.getElementById('lightbox-modal');
  if (modal) modal.classList.add('hidden');
}

function resetLightboxZoom() {
  lightboxState.scale = 1.0;
  lightboxState.translateX = 0;
  lightboxState.translateY = 0;
  applyLightboxTransform();
}

function setLightboxActualSize() {
  lightboxState.scale = 1.5;
  lightboxState.translateX = 0;
  lightboxState.translateY = 0;
  applyLightboxTransform();
}

function zoomLightbox(factor) {
  const newScale = Math.max(lightboxState.minScale, Math.min(lightboxState.maxScale, lightboxState.scale * factor));
  lightboxState.scale = newScale;
  applyLightboxTransform();
}

function zoomLightboxAt(factor, mouseX, mouseY) {
  const oldScale = lightboxState.scale;
  const newScale = Math.max(lightboxState.minScale, Math.min(lightboxState.maxScale, oldScale * factor));
  if (Math.abs(newScale - oldScale) < 1e-4) return;

  const scaleRatio = newScale / oldScale;
  lightboxState.translateX = mouseX - (mouseX - lightboxState.translateX) * scaleRatio;
  lightboxState.translateY = mouseY - (mouseY - lightboxState.translateY) * scaleRatio;
  lightboxState.scale = newScale;

  applyLightboxTransform();
}

function applyLightboxTransform() {
  const canvas = document.getElementById('lightbox-canvas');
  const statusEl = document.getElementById('lightbox-status');
  if (canvas) {
    canvas.style.transform = `translate(${lightboxState.translateX}px, ${lightboxState.translateY}px) scale(${lightboxState.scale})`;
  }
  if (statusEl) {
    statusEl.innerText = `${Math.round(lightboxState.scale * 100)}%`;
  }
}

