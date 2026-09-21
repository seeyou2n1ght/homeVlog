// HomeVlog Studio Audit Console 2.0 Client Interaction Engine

// 全局应用状态
const state = {
  currentTab: 'anomalies', // 'anomalies' | 'tree'
  currentCategory: 'all',  // 'all' | 'fp_suspect' | 'fn_suspect' | 'jitter' | 'reviewed'
  sortMode: 'energy',      // 'energy' | 'time_asc' | 'time_desc' | 'duration'
  searchQuery: '',
  filterDate: '',
  filterCam: '',
  selectedScenario: '',
  selectedAnomalyIndex: -1,
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

  // BBox 标注状态
  bboxMode: false,
  currentBBoxes: [], // [{class_id: 0, x_center, y_center, width, height}]
  isDrawingBBox: false,
  drawStart: { x: 0, y: 0 },
  
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
  initBBoxCanvas();
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

  // 日期与机位下拉过滤
  const dateSel = document.getElementById('filter-date-select');
  if (dateSel) {
    dateSel.addEventListener('change', (e) => {
      state.filterDate = e.target.value;
      loadAnomalies();
    });
  }
  const camSel = document.getElementById('filter-cam-select');
  if (camSel) {
    camSel.addEventListener('change', (e) => {
      state.filterCam = e.target.value;
      loadAnomalies();
    });
  }

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
  if (btnZoomIn) btnZoomIn.addEventListener('click', () => zoomTimeline(0.75));
  const btnZoomOut = document.getElementById('btn-zoom-out');
  if (btnZoomOut) btnZoomOut.addEventListener('click', () => zoomTimeline(1.33));
  const btnZoomFocus = document.getElementById('btn-zoom-focus');
  if (btnZoomFocus) btnZoomFocus.addEventListener('click', () => focusCurrentSegment());
  const btnZoomReset = document.getElementById('btn-zoom-reset');
  if (btnZoomReset) btnZoomReset.addEventListener('click', () => resetTimelineZoom());

  // 快捷切片导航按钮 (修复 K 为上一切片，J 为下一切片)
  const btnPrevSeg = document.getElementById('btn-prev-seg');
  if (btnPrevSeg) btnPrevSeg.addEventListener('click', () => selectNextSegment(-1));
  const btnNextSeg = document.getElementById('btn-next-seg');
  if (btnNextSeg) btnNextSeg.addEventListener('click', () => selectNextSegment(1));

  // 场景归因标签点击切换
  document.querySelectorAll('.scenario-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const sc = chip.getAttribute('data-scenario');
      if (state.selectedScenario === sc) {
        state.selectedScenario = '';
        chip.classList.remove('active');
      } else {
        document.querySelectorAll('.scenario-chip').forEach(c => c.classList.remove('active'));
        state.selectedScenario = sc;
        chip.classList.add('active');
      }
    });
  });

  // 撤销打标按钮
  const btnClearReview = document.getElementById('btn-clear-review');
  if (btnClearReview) btnClearReview.addEventListener('click', clearCurrentReview);

  // 四路决策裁决按钮
  const btnLabelTp = document.getElementById('btn-label-tp');
  if (btnLabelTp) btnLabelTp.addEventListener('click', () => submitCurrentReview('CONFIRMED_MOTION'));
  const btnLabelFp = document.getElementById('btn-label-fp');
  if (btnLabelFp) btnLabelFp.addEventListener('click', () => submitCurrentReview('FALSE_ALARM'));
  const btnLabelFn = document.getElementById('btn-label-fn');
  if (btnLabelFn) btnLabelFn.addEventListener('click', () => submitCurrentReview('MISSED_MOTION'));
  const btnLabelTn = document.getElementById('btn-label-tn');
  if (btnLabelTn) btnLabelTn.addEventListener('click', () => submitCurrentReview('CONFIRMED_STATIC'));

  // 视觉强化与预览按钮
  const btnYolo = document.getElementById('btn-yolo-enhance');
  if (btnYolo) btnYolo.addEventListener('click', runYoloEnhance);
  const btnClip = document.getElementById('btn-clip-preview');
  if (btnClip) btnClip.addEventListener('click', showClipPreview);

  // BBox 标注工具栏按钮
  const btnToggleBBox = document.getElementById('btn-toggle-bbox');
  if (btnToggleBBox) btnToggleBBox.addEventListener('click', toggleBBoxMode);
  const btnBBoxUndo = document.getElementById('btn-bbox-undo');
  if (btnBBoxUndo) btnBBoxUndo.addEventListener('click', undoBBox);
  const btnBBoxSave = document.getElementById('btn-bbox-save');
  if (btnBBoxSave) btnBBoxSave.addEventListener('click', saveCurrentBBoxes);

  // 全量导出报表下载
  const btnExportCsv = document.getElementById('btn-export-csv');
  if (btnExportCsv) btnExportCsv.addEventListener('click', () => triggerReportDownload('csv'));
  const btnExportJson = document.getElementById('btn-export-json');
  if (btnExportJson) btnExportJson.addEventListener('click', () => triggerReportDownload('json'));

  // 真实反馈帧归档弹窗
  const btnOpenArchive = document.getElementById('btn-open-archive-modal');
  if (btnOpenArchive) btnOpenArchive.addEventListener('click', openArchiveModal);
  const btnArchiveClose = document.getElementById('btn-archive-close');
  if (btnArchiveClose) btnArchiveClose.addEventListener('click', closeArchiveModal);
  const btnArchiveModalClose = document.getElementById('btn-archive-modal-close');
  if (btnArchiveModalClose) btnArchiveModalClose.addEventListener('click', closeArchiveModal);
  const btnArchiveBatchRun = document.getElementById('btn-archive-batch-run');
  if (btnArchiveBatchRun) btnArchiveBatchRun.addEventListener('click', triggerBatchArchive);

  // YOLO 数据集导出模态框事件
  const btnOpenYolo = document.getElementById('btn-open-yolo-modal');
  if (btnOpenYolo) btnOpenYolo.addEventListener('click', openYoloModal);
  const btnYoloClose = document.getElementById('btn-yolo-modal-close');
  if (btnYoloClose) btnYoloClose.addEventListener('click', closeYoloModal);
  const btnYoloCancel = document.getElementById('btn-yolo-modal-cancel');
  if (btnYoloCancel) btnYoloCancel.addEventListener('click', closeYoloModal);
  const btnYoloDoExport = document.getElementById('btn-yolo-do-export');
  if (btnYoloDoExport) btnYoloDoExport.addEventListener('click', exportYoloDataset);

  // 调参建议模态框事件
  const btnOpenTuning = document.getElementById('btn-open-tuning-modal');
  if (btnOpenTuning) btnOpenTuning.addEventListener('click', openTuningModal);
  const btnTuningClose = document.getElementById('btn-tuning-modal-close');
  if (btnTuningClose) btnTuningClose.addEventListener('click', closeTuningModal);
  const btnTuningCancel = document.getElementById('btn-tuning-modal-cancel');
  if (btnTuningCancel) btnTuningCancel.addEventListener('click', closeTuningModal);

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

  // 高清视窗开启时优先响应放大与 Esc
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
  } else if (e.key === 'j' || e.key === 'J' || e.key === 'ArrowDown') {
    selectNextSegment(1);
  } else if (e.key === 'k' || e.key === 'K' || e.key === 'ArrowUp') {
    selectNextSegment(-1);
  } else if (e.key === 'b' || e.key === 'B') {
    toggleBBoxMode();
  } else if (e.key === 'u' || e.key === 'U') {
    clearCurrentReview();
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
    const yModal = document.getElementById('yolo-dataset-modal');
    if (yModal && !yModal.classList.contains('hidden')) closeYoloModal();
    const tModal = document.getElementById('tuning-insights-modal');
    if (tModal && !tModal.classList.contains('hidden')) closeTuningModal();
    if (lightboxState && lightboxState.isOpen) closeLightbox();
    if (state.bboxMode) toggleBBoxMode();
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

// 触发报表下载
function triggerReportDownload(fmt) {
  showToast(`正在导出并下载 ${fmt.toUpperCase()} 复核报表...`, 'info');
  const a = document.createElement('a');
  a.href = `/api/export?format=${fmt}`;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => a.remove(), 200);
}

// ================== 切片本地相对时间换算核心算法 ==================
function getSegmentLocalRange(seg) {
  const fileDur = state.timeline.fileDuration || 300;
  const rawStart = Number(seg.start_time || 0);
  const rawEnd = Number(seg.end_time || rawStart);
  const dur = Number(seg.duration || (rawEnd - rawStart));
  
  // 1. 首选数据库显式存储的 file_start_offset
  if (seg.file_start_offset != null && !isNaN(Number(seg.file_start_offset))) {
    const off = Number(seg.file_start_offset);
    if (rawStart >= off) {
      const ls = rawStart - off;
      return { start: Math.max(0, ls), dur: dur, end: Math.min(fileDur, ls + dur) };
    }
  }

  // 2. 次选解析出的视频起始绝对秒数 fileOffsetSec
  const fileOff = state.timeline.fileOffsetSec || 0;
  if (fileOff > 0 && rawStart >= fileOff) {
    const ls = rawStart - fileOff;
    return { start: Math.max(0, ls), dur: dur, end: Math.min(fileDur, ls + dur) };
  }

  // 3. 若 rawStart 大于文件总时长，说明是日内绝对秒数，减去首切片起始秒数对齐
  const firstStart = (state.currentSegments && state.currentSegments.length > 0) ? Number(state.currentSegments[0].start_time || 0) : 0;
  if (rawStart > fileDur && firstStart > 0 && rawStart >= firstStart) {
    const ls = rawStart - firstStart;
    return { start: Math.max(0, ls), dur: dur, end: Math.min(fileDur, ls + dur) };
  }

  return { start: Math.max(0, rawStart), dur: dur, end: Math.min(fileDur, rawStart + dur) };
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
    if (precRecEl) {
      precRecEl.innerText = `${data.metrics.precision == null ? "N/A" : data.metrics.precision + "%"} / ${data.metrics.recall == null ? "N/A" : data.metrics.recall + "%"}`;
    }

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
      } else if (['DYNAMIC', 'DYNAMIC_AUDIO', 'PRESENCE'].includes(item.state) && (item.avg_confidence === 0 || item.max_energy < 8.0)) {
        fpCount++;
      } else if (['STATIC', 'NIGHT_STATIONARY'].includes(item.state) && item.max_energy >= 1.5) {
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
    let url = `/api/anomalies?category=${state.currentCategory}&limit=100`;
    if (state.filterDate) url += `&date=${encodeURIComponent(state.filterDate)}`;
    if (state.filterCam) url += `&cam_index=${encodeURIComponent(state.filterCam)}`;

    const res = await fetch(url);
    const list = await res.json();
    state.anomalies = list || [];
    
    const countEl = document.getElementById('anomaly-count');
    if (countEl) countEl.innerText = state.anomalies.length;

    // 动态提取日期与机位填充下拉列表
    populateFilterDropdowns();

    applyFilterAndSort();

    // 默认聚焦第一条
    if (state.filteredAnomalies.length > 0) {
      selectAnomalyItem(0);
    }
  } catch (err) {
    console.error('Failed to load anomalies:', err);
    if (container) container.innerHTML = '<div class="state-loader error">加载争议样本失败，请检查服务状态</div>';
  }
}

function populateFilterDropdowns() {
  const dateSel = document.getElementById('filter-date-select');
  const camSel = document.getElementById('filter-cam-select');
  if (!dateSel || !camSel) return;

  const dates = new Set();
  const cams = new Set();

  state.anomalies.forEach(item => {
    if (item.date) dates.add(item.date);
    if (item.cam_index != null) cams.add(item.cam_index);
  });

  const curDate = state.filterDate;
  const curCam = state.filterCam;

  dateSel.innerHTML = '<option value="">📅 所有日期</option>';
  Array.from(dates).sort().reverse().forEach(d => {
    const opt = document.createElement('option');
    opt.value = d;
    opt.innerText = `📅 ${d}`;
    if (d === curDate) opt.selected = true;
    dateSel.appendChild(opt);
  });

  camSel.innerHTML = '<option value="">📷 所有机位</option>';
  Array.from(cams).sort().forEach(c => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.innerText = `📷 机位 ${c}`;
    if (c.toString() === curCam.toString()) opt.selected = true;
    camSel.appendChild(opt);
  });
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
      const sc = (item.scenario || '').toLowerCase();
      return fn.includes(q) || eng.includes(q) || st.includes(q) || reason.includes(q) || sc.includes(q);
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

  state.filteredAnomalies.forEach((item, index) => {
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
    } else if (item.anomaly_type === 'fp_suspect' || item.anomaly_type === 'FP_SUSPECT') {
      reason = '疑似光影假阳性 (无目标置信度)';
      icon = '⚠️';
      badgeClass = 'chip-fp';
    } else if (item.anomaly_type === 'fn_suspect' || item.anomaly_type === 'FN_SUSPECT') {
      reason = '疑似微动作漏判 (能量临界)';
      icon = '🎯';
      badgeClass = 'chip-fn';
    } else if (item.duration < 3.0) {
      reason = '极短碎片跳变';
      icon = '⏱️';
      badgeClass = 'chip-jitter';
    }

    const fn = (item.filepath || '').split('\\').pop().split('/').pop();
    const durSec = (item.duration || 0).toFixed(1);
    const engVal = (item.max_energy || 0).toFixed(1);
    const scTag = item.scenario ? `<span class="scenario-badge-mini">${item.scenario}</span>` : '';

    div.innerHTML = `
      <div class="anomaly-card-top">
        <span class="anomaly-id font-mono">#${item.id} • ${item.state}</span>
        <span class="anomaly-dur font-mono">${durSec}s</span>
      </div>
      <div class="anomaly-reason-tag ${badgeClass}">
        <span>${icon}</span>
        <span>${reason}</span>
        ${scTag}
      </div>
      <div class="anomaly-card-footer font-mono">
        <span>⚡ 峰值: ${engVal}</span>
        <span class="file-ellipsis" title="${item.filepath}">${fn}</span>
      </div>
    `;

    div.addEventListener('click', () => {
      selectAnomalyItem(index);
    });

    container.appendChild(div);
  });
}

function selectAnomalyItem(index) {
  if (index < 0 || index >= state.filteredAnomalies.length) return;
  state.selectedAnomalyIndex = index;
  const item = state.filteredAnomalies[index];

  document.querySelectorAll('.anomaly-card').forEach(el => el.classList.remove('selected'));
  const card = document.getElementById(`anomaly-item-${item.id}`);
  if (card) {
    card.classList.add('selected');
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  loadFileSegments(item.file_id, item.filepath, item.id);
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

// 解析文件名中的真实墙钟时间
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

function formatClockTime(absSec) {
  const totalSec = Math.floor(absSec) % 86400;
  const hh = Math.floor(totalSec / 3600);
  const mm = Math.floor((totalSec % 3600) / 60);
  const ss = Math.floor(totalSec % 60);
  return `${hh.toString().padStart(2, '0')}:${mm.toString().padStart(2, '0')}:${ss.toString().padStart(2, '0')}`;
}

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

    const range = getSegmentLocalRange(s);
    const leftPct = (range.start / totalDur) * 100;
    const widthPct = (range.dur / totalDur) * 100;

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

  segs.forEach((s, idx) => {
    const range = getSegmentLocalRange(s);
    const localStart = range.start;
    const localDur = range.dur;
    const localEnd = range.end;

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
  const range = getSegmentLocalRange(seg);
  const localMid = (range.start + range.end) / 2.0;

  const targetSpan = Math.min(state.timeline.fileDuration, Math.max(16.0, range.dur * 4.5));
  
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

  const minSpan = 4.0;
  const maxSpan = totalDur;
  newSpan = Math.max(minSpan, Math.min(maxSpan, newSpan));

  const anchorSec = state.timeline.viewStart + (curSpan * anchorRatio);
  let newStart = anchorSec - (newSpan * anchorRatio);
  let newEnd = newStart + newSpan;

  if (newStart < 0) {
    newStart = 0;
    newEnd = newSpan;
  } else if (newEnd > totalDur) {
    newEnd = totalDur;
    newStart = Math.max(0, totalDur - newSpan);
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

  // 恢复切片的场景归因与用户备注
  state.selectedScenario = seg.scenario || '';
  document.querySelectorAll('.scenario-chip').forEach(chip => {
    if (chip.getAttribute('data-scenario') === state.selectedScenario) {
      chip.classList.add('active');
    } else {
      chip.classList.remove('active');
    }
  });
  const notesInput = document.getElementById('review-notes-input');
  if (notesInput) notesInput.value = seg.user_notes || '';

  // 重置 BBox 标注状态
  resetBBoxes();

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
    if (seg.scenario) {
      const scChip = document.createElement('span');
      scChip.className = 'chip chip-scenario';
      scChip.innerText = `🏷️ ${seg.scenario}`;
      tagsWrap.appendChild(scChip);
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
  const range = getSegmentLocalRange(seg);
  const relStart = range.start;
  const relEnd = range.end;
  const relMid = (relStart + relEnd) / 2.0;

  const tStart = seg.start_time;
  const tMid = (seg.start_time + seg.end_time) / 2.0;
  const tEnd = Math.max(seg.end_time - 0.1, seg.start_time);

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
  setSafeFrameImage('mid', fp, tMid, () => {
    // 中继帧加载完成后，如在标注模式则重绘 Canvas
    if (state.bboxMode) syncBBoxCanvasSize();
  });
  setSafeFrameImage('end', fp, tEnd);
}

function setElementText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}

function setSafeFrameImage(pos, filepath, timestamp, onLoadCallback) {
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
    if (onLoadCallback) onLoadCallback();
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

// ================== BBox 交互式目标画框标注引擎 ==================

function initBBoxCanvas() {
  const canvas = document.getElementById('bbox-canvas');
  if (!canvas) return;

  canvas.addEventListener('mousedown', (e) => {
    if (!state.bboxMode) return;
    const rect = canvas.getBoundingClientRect();
    state.isDrawingBBox = true;
    state.drawStart = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!state.bboxMode || !state.isDrawingBBox) return;
    const rect = canvas.getBoundingClientRect();
    const curX = e.clientX - rect.left;
    const curY = e.clientY - rect.top;

    drawAllBBoxes(curX, curY);
  });

  window.addEventListener('mouseup', (e) => {
    if (!state.bboxMode || !state.isDrawingBBox) return;
    state.isDrawingBBox = false;

    const rect = canvas.getBoundingClientRect();
    const curX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const curY = Math.max(0, Math.min(rect.height, e.clientY - rect.top));

    const x1 = Math.min(state.drawStart.x, curX);
    const x2 = Math.max(state.drawStart.x, curX);
    const y1 = Math.min(state.drawStart.y, curY);
    const y2 = Math.max(state.drawStart.y, curY);

    const w = x2 - x1;
    const h = y2 - y1;

    // 过滤无意义点击或超小噪点框
    if (w > 10 && h > 10 && rect.width > 0 && rect.height > 0) {
      const xc = (x1 + w / 2) / rect.width;
      const yc = (y1 + h / 2) / rect.height;
      const nw = w / rect.width;
      const nh = h / rect.height;

      state.currentBBoxes.push({
        class_id: 0, // 0: person
        x_center: xc,
        y_center: yc,
        width: nw,
        height: nh
      });
      showToast(`已添加目标框 #${state.currentBBoxes.length} (person)`, 'info');
    }

    drawAllBBoxes();
  });
}

function toggleBBoxMode() {
  state.bboxMode = !state.bboxMode;
  const canvas = document.getElementById('bbox-canvas');
  const toolbar = document.getElementById('bbox-toolbar');
  const btnToggle = document.getElementById('btn-toggle-bbox');

  if (state.bboxMode) {
    if (canvas) canvas.classList.remove('hidden');
    if (toolbar) toolbar.classList.remove('hidden');
    if (btnToggle) {
      btnToggle.innerText = '✕ 退出标注 (B)';
      btnToggle.classList.add('active');
    }
    syncBBoxCanvasSize();
    drawAllBBoxes();
    showToast('已进入目标框选标注模式：在峰值画面拖拽鼠标框选漏检人物', 'info');
  } else {
    if (canvas) canvas.classList.add('hidden');
    if (toolbar) toolbar.classList.add('hidden');
    if (btnToggle) {
      btnToggle.innerText = '✏️ 标注 (B)';
      btnToggle.classList.remove('active');
    }
  }
}

function syncBBoxCanvasSize() {
  const canvas = document.getElementById('bbox-canvas');
  const img = document.getElementById('img-mid');
  if (!canvas || !img) return;

  canvas.width = img.clientWidth || 480;
  canvas.height = img.clientHeight || 270;
}

function drawAllBBoxes(dragX = null, dragY = null) {
  const canvas = document.getElementById('bbox-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const cw = canvas.width;
  const ch = canvas.height;

  // 1. 绘制已固化的 BBox
  state.currentBBoxes.forEach((b, idx) => {
    const x = (b.x_center - b.width / 2) * cw;
    const y = (b.y_center - b.height / 2) * ch;
    const w = b.width * cw;
    const h = b.height * ch;

    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
    ctx.fillRect(x, y, w, h);

    ctx.fillStyle = '#10b981';
    ctx.font = '11px monospace';
    ctx.fillText(`person #${idx + 1}`, x + 4, Math.max(y - 4, 12));
  });

  // 2. 绘制当前拖拽中的临时虚线框
  if (state.isDrawingBBox && dragX !== null && dragY !== null) {
    const x1 = Math.min(state.drawStart.x, dragX);
    const y1 = Math.min(state.drawStart.y, dragY);
    const w = Math.abs(dragX - state.drawStart.x);
    const h = Math.abs(dragY - state.drawStart.y);

    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, w, h);
    ctx.fillStyle = 'rgba(6, 182, 212, 0.15)';
    ctx.fillRect(x1, y1, w, h);
    ctx.setLineDash([]);
  }
}

function undoBBox() {
  if (state.currentBBoxes.length > 0) {
    state.currentBBoxes.pop();
    drawAllBBoxes();
    showToast('已撤销上一个目标框', 'info');
  }
}

function resetBBoxes() {
  state.currentBBoxes = [];
  const canvas = document.getElementById('bbox-canvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

async function saveCurrentBBoxes() {
  if (state.selectedSegIndex < 0 || state.currentBBoxes.length === 0) {
    showToast('暂无绘制的目标边框', 'info');
    return;
  }
  const seg = state.currentSegments[state.selectedSegIndex];

  try {
    const res = await fetch('/api/save_bbox', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        segment_id: seg.id,
        boxes: state.currentBBoxes
      })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`已成功保存 ${data.box_count} 个目标标注至 YOLO 侧边文件`, 'success');
      toggleBBoxMode();
    } else {
      showToast('保存标注失败', 'error');
    }
  } catch (err) {
    console.error('Failed to save BBoxes:', err);
    showToast('保存标注请求异常', 'error');
  }
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

// ================== 裁决打标、撤销与自动步进 ==================

async function submitCurrentReview(manualLabel) {
  if (state.selectedSegIndex < 0) return;
  const seg = state.currentSegments[state.selectedSegIndex];
  const notesInput = document.getElementById('review-notes-input');
  const userNotes = notesInput ? notesInput.value.trim() : '';
  const scenario = state.selectedScenario || '';

  try {
    const res = await fetch('/api/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        segment_id: seg.id,
        manual_label: manualLabel,
        notes: userNotes,
        scenario: scenario
      })
    });
    const data = await res.json();
    if (data.success) {
      seg.manual_label = manualLabel;
      seg.scenario = scenario;
      seg.user_notes = userNotes;

      // 若在当前切片上手动绘制了 BBox，伴随一并自动保存为侧边标注
      if (state.currentBBoxes.length > 0) {
        saveCurrentBBoxes();
      }

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
      const extraSc = scenario ? ` • 归因: [${scenario}]` : '';
      showToast(`已标记为 [${labelName}]${extraSc} 并同步归档代表帧`, 'success');

      selectNextSegment(1);
    }
  } catch (err) {
    console.error('Submit review failed:', err);
    showToast('提交打标失败', 'error');
  }
}

async function clearCurrentReview() {
  if (state.selectedSegIndex < 0) return;
  const seg = state.currentSegments[state.selectedSegIndex];
  if (!seg.manual_label) {
    showToast('当前切片尚未打标，无需撤销', 'info');
    return;
  }

  try {
    const res = await fetch('/api/clear_review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ segment_id: seg.id })
    });
    const data = await res.json();
    if (data.success) {
      seg.manual_label = null;
      seg.scenario = '';
      seg.user_notes = '';
      state.selectedScenario = '';
      document.querySelectorAll('.scenario-chip').forEach(c => c.classList.remove('active'));
      const notesInput = document.getElementById('review-notes-input');
      if (notesInput) notesInput.value = '';

      renderDualTimeline();
      loadOverview();
      loadCategoryBadges();
      loadArchiveStats();
      showToast(`已成功撤销切片 #${seg.id} 的复核标注`, 'success');
    } else {
      showToast('撤销打标失败', 'error');
    }
  } catch (err) {
    console.error('Failed to clear review:', err);
    showToast('撤销打标请求异常', 'error');
  }
}

function selectNextSegment(step) {
  // 若在疑难排查模式下，优先在争议队列中连续步进
  if (state.currentTab === 'anomalies') {
    const nextAno = (state.selectedAnomalyIndex >= 0 ? state.selectedAnomalyIndex : 0) + step;
    if (nextAno >= 0 && nextAno < state.filteredAnomalies.length) {
      selectAnomalyItem(nextAno);
      return;
    }
  }
  // 否则在当前文件切片列表中步进
  const nextIdx = state.selectedSegIndex + step;
  if (nextIdx >= 0 && nextIdx < state.currentSegments.length) {
    selectSegment(nextIdx, true);
  }
}

// ================== YOLO 数据集导出模态框 ==================

function openYoloModal() {
  const modal = document.getElementById('yolo-dataset-modal');
  const resBox = document.getElementById('yolo-export-result');
  if (resBox) resBox.classList.add('hidden');
  if (modal) modal.classList.remove('hidden');
}

function closeYoloModal() {
  const modal = document.getElementById('yolo-dataset-modal');
  if (modal) modal.classList.add('hidden');
}

async function exportYoloDataset() {
  const btn = document.getElementById('btn-yolo-do-export');
  const valRatioSel = document.getElementById('yolo-val-ratio');
  const valRatio = valRatioSel ? parseFloat(valRatioSel.value) : 0.2;

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spin-ring" style="width: 14px; height: 14px; display: inline-block;"></span> 正在构建数据集...';
  }

  try {
    const res = await fetch('/api/export_yolo_dataset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ val_ratio: valRatio })
    });
    const data = await res.json();
    if (data.success) {
      const resBox = document.getElementById('yolo-export-result');
      const statsEl = document.getElementById('yolo-export-stats');
      const cmdEl = document.getElementById('yolo-train-cmd');

      if (statsEl && data.stats) {
        statsEl.innerHTML = `
          <div>📁 训练集: <strong>${data.stats.train_images}</strong> 张图片 | <strong>${data.stats.train_boxes}</strong> 个目标标注</div>
          <div>📁 验证集: <strong>${data.stats.val_images}</strong> 张图片 | <strong>${data.stats.val_boxes}</strong> 个目标标注</div>
          <div>🛡️ 困难负样本: <strong>${data.stats.negatives}</strong> 张 (空 txt 标注，抑制虚警)</div>
          <div>📄 配置文件: <span class="font-mono" style="color: var(--color-cyan);">${data.yaml_path}</span></div>
        `;
      }
      if (cmdEl && data.train_cmd) {
        cmdEl.innerText = data.train_cmd;
        cmdEl.onclick = () => {
          navigator.clipboard.writeText(data.train_cmd);
          showToast('已复制训练命令至剪贴板！', 'success');
        };
      }
      if (resBox) resBox.classList.remove('hidden');
      showToast('YOLO 数据集导出完成！', 'success');
    } else {
      showToast(data.message || '导出数据集失败', 'error');
    }
  } catch (err) {
    console.error('YOLO dataset export error:', err);
    showToast('导出数据集请求异常', 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '⚡ 立即导出数据集';
    }
  }
}

// ================== 调参建议模态框 ==================

async function openTuningModal() {
  const modal = document.getElementById('tuning-insights-modal');
  const sumBox = document.getElementById('tuning-summary-box');
  const listEl = document.getElementById('tuning-recommendations-list');

  if (modal) modal.classList.remove('hidden');
  if (sumBox) sumBox.innerHTML = '<div class="state-loader">正在全量计算误报/漏报归因与能量分位点...</div>';
  if (listEl) listEl.innerHTML = '';

  try {
    const res = await fetch('/api/tuning_insights');
    const data = await res.json();

    if (sumBox) {
      const dist = data.distribution || {};
      const sc = data.scenario_counts || {};
      const nrg = data.energy_stats || {};

      let scHtml = Object.entries(sc).map(([k, v]) => `<span>🏷️ ${k}: <strong>${v}</strong></span>`).join(' • ') || '暂无场景标注';

      sumBox.innerHTML = `
        <div class="tuning-metric-pill">
          <span class="tuning-metric-label">已复核切片</span>
          <span class="tuning-metric-val font-mono">${data.total_reviewed} 处</span>
        </div>
        <div class="tuning-metric-pill">
          <span class="tuning-metric-label">误报 (FP)</span>
          <span class="tuning-metric-val font-mono" style="color: var(--color-red);">${dist.fp || 0} 处 (均能 ${nrg.fp_avg_energy || 0})</span>
        </div>
        <div class="tuning-metric-pill">
          <span class="tuning-metric-label">漏报 (FN)</span>
          <span class="tuning-metric-val font-mono" style="color: var(--color-blue);">${dist.fn || 0} 处 (均能 ${nrg.fn_avg_energy || 0})</span>
        </div>
        <div class="tuning-metric-pill" style="flex: 2;">
          <span class="tuning-metric-label">场景归因分布</span>
          <span class="tuning-metric-val" style="font-size: 12px; font-weight: normal; color: var(--text-secondary); margin-top: 4px;">${scHtml}</span>
        </div>
      `;
    }

    if (listEl) {
      const recs = data.recommendations || [];
      if (recs.length === 0) {
        listEl.innerHTML = `
          <div class="state-loader" style="text-align: left; padding: 16px;">
            当前审核样本量尚不足或分类指标极佳，未触发规则阈值微调报警。<br>
            建议继续复核 5~10 处疑难切片并打标「光影刚性」或「婴儿微动」场景。
          </div>
        `;
      } else {
        recs.forEach(r => {
          const card = document.createElement('div');
          card.className = 'tuning-rec-card';
          card.innerHTML = `
            <div class="tuning-rec-header">
              <span class="tuning-rec-cat">${r.category}</span>
              <span class="tuning-rec-param">${r.param}</span>
              <span class="tuning-rec-change font-mono">${r.current_value} ➔ <strong style="color: var(--color-green);">${r.suggested_value}</strong></span>
            </div>
            <div class="tuning-rec-rationale">${r.rationale}</div>
          `;
          listEl.appendChild(card);
        });
      }
    }
  } catch (err) {
    console.error('Failed to load tuning insights:', err);
    if (sumBox) sumBox.innerHTML = '<div class="state-loader error">加载调参建议失败</div>';
  }
}

function closeTuningModal() {
  const modal = document.getElementById('tuning-insights-modal');
  if (modal) modal.classList.add('hidden');
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
    const st = await res.json();
    applyRerenderState(st);
  } catch (err) {
    console.warn('Status poll failed:', err);
  }
}

async function startRerender() {
  const { date, cam } = getActiveDateAndCam();
  const btnStart = document.getElementById('btn-rerender-start');
  if (btnStart) btnStart.disabled = true;

  try {
    const res = await fetch('/api/rerender', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ date, cam_index: cam, version: 'reviewed' })
    });
    const data = await res.json();
    if (data.error) {
      showToast(data.error, 'error');
      if (btnStart) btnStart.disabled = false;
      return;
    }

    showToast('重渲染任务已提交执行', 'info');
    const btnCancel = document.getElementById('btn-rerender-cancel');
    if (btnCancel) btnCancel.innerText = '中断任务';

    if (rerenderPollTimer) clearInterval(rerenderPollTimer);
    rerenderPollTimer = setInterval(() => checkCurrentRerenderStatus(date, cam), 1000);
  } catch (err) {
    console.error('Failed to start rerender:', err);
    showToast('启动重渲染失败', 'error');
    if (btnStart) btnStart.disabled = false;
  }
}

function applyRerenderState(st) {
  if (!st || st.status === 'IDLE') return;

  const status = st.status;
  const statusEl = document.getElementById('rerender-file-check-status');
  const stageEl = document.getElementById('rerender-stage-text');
  const pctEl = document.getElementById('rerender-pct-text');
  const batchEl = document.getElementById('rerender-batch-text');
  const timeEl = document.getElementById('rerender-time-text');
  const fill = document.getElementById('rerender-progress-fill');
  const btnCancel = document.getElementById('btn-rerender-cancel');

  if (st.missing_files && st.missing_files.length > 0) {
    const alertEl = document.getElementById('rerender-missing-alert');
    const listEl = document.getElementById('rerender-missing-list');
    if (alertEl) alertEl.classList.remove('hidden');
    if (listEl) {
      listEl.innerHTML = st.missing_files.map(f => `<div>• ${f}</div>`).join('');
    }
  }

  if (timeEl && st.elapsed_s != null) {
    timeEl.innerText = `已耗时: ${st.elapsed_s}s`;
  }

  if (fill && st.progress != null) {
    fill.style.width = `${st.progress}%`;
  }
  if (pctEl && st.progress != null) {
    pctEl.innerText = `${st.progress}%`;
  }

  if (batchEl && st.total_batches != null) {
    batchEl.innerText = `批次: ${st.current_batch || 0} / ${st.total_batches}`;
  }

  if (status === 'CHECKING') {
    if (statusEl) {
      statusEl.className = 'status-chip';
      statusEl.innerText = '正在探测素材可达性...';
    }
    if (stageEl) stageEl.innerText = '前置存储物理探测';
    if (btnCancel) btnCancel.innerText = '中断任务';
  } else if (status === 'RENDERING') {
    if (statusEl) {
      statusEl.className = 'status-chip chip-reviewed';
      statusEl.innerText = '物理校验通过';
    }
    if (stageEl) stageEl.innerText = `硬件批次渲染中 (${st.current_batch}/${st.total_batches})`;
    if (btnCancel) btnCancel.innerText = '中断任务';
  } else if (status === 'CONCATING') {
    if (stageEl) stageEl.innerText = '合并最终 Vlog 成片并生成配套资产...';
    if (btnCancel) btnCancel.innerText = '中断任务';
  } else if (status === 'COMPLETED') {
    if (stageEl) stageEl.innerText = `✅ 重渲染完成！体积: ${st.file_size_mb || 0} MB`;
    if (fill) fill.style.background = 'var(--color-green)';
    if (btnCancel) btnCancel.innerText = '关闭';
    const btnStart = document.getElementById('btn-rerender-start');
    if (btnStart) btnStart.classList.add('hidden');
    
    currentRerenderOutputFile = st.output_file || '';
    const btnOpenDir = document.getElementById('btn-rerender-open-dir');
    if (btnOpenDir && currentRerenderOutputFile) {
      btnOpenDir.classList.remove('hidden');
    }

    if (rerenderPollTimer) {
      clearInterval(rerenderPollTimer);
      rerenderPollTimer = null;
    }
    showToast('🎉 Vlog 即时重新浓缩压制完成！', 'success');
  } else if (status === 'FAILED') {
    if (stageEl) stageEl.innerText = `❌ 渲染失败: ${st.error || '未知错误'}`;
    if (fill) fill.style.background = 'var(--color-red)';
    if (btnCancel) btnCancel.innerText = '关闭';
    const btnStart = document.getElementById('btn-rerender-start');
    if (btnStart) {
      btnStart.disabled = false;
      btnStart.classList.remove('hidden');
    }
    if (rerenderPollTimer) {
      clearInterval(rerenderPollTimer);
      rerenderPollTimer = null;
    }
  } else if (status === 'CANCELLED') {
    if (stageEl) stageEl.innerText = '⚠️ 任务已被用户中断';
    if (btnCancel) btnCancel.innerText = '关闭';
    const btnStart = document.getElementById('btn-rerender-start');
    if (btnStart) {
      btnStart.disabled = false;
      btnStart.classList.remove('hidden');
    }
    if (rerenderPollTimer) {
      clearInterval(rerenderPollTimer);
      rerenderPollTimer = null;
    }
  }
}

async function openRerenderOutputDir() {
  if (!currentRerenderOutputFile) return;
  try {
    const res = await fetch(`/api/open_output_dir?filepath=${encodeURIComponent(currentRerenderOutputFile)}`);
    const data = await res.json();
    if (data.success) {
      showToast('已在资源管理器中定位成片文件', 'success');
    } else {
      showToast('定位文件失败', 'error');
    }
  } catch (err) {
    console.error('Failed to open dir:', err);
  }
}

// ================== 反馈帧归档状态管理 ==================

async function loadArchiveStats() {
  try {
    const res = await fetch('/api/archive_stats');
    const data = await res.json();
    
    const countBadge = document.getElementById('archive-count-badge');
    if (countBadge) countBadge.innerText = data.total_images || 0;

    const dirEl = document.getElementById('archive-dir-path');
    if (dirEl) dirEl.innerText = data.archive_dir || 'data/feedback_archive';

    const totImgEl = document.getElementById('archive-total-images');
    if (totImgEl) totImgEl.innerText = `${data.total_images || 0} 张`;

    const totSizeEl = document.getElementById('archive-total-size');
    if (totSizeEl) totSizeEl.innerText = `${data.total_size_mb || 0} MB`;

    const distWrap = document.getElementById('archive-label-breakdown');
    if (distWrap && data.labels) {
      distWrap.innerHTML = '';
      Object.entries(data.labels).forEach(([lbl, cnt]) => {
        const span = document.createElement('span');
        span.className = 'chip font-mono';
        span.style.fontSize = '11px';
        span.innerText = `${lbl}: ${cnt}`;
        distWrap.appendChild(span);
      });
    }
  } catch (err) {
    console.warn('Failed to load archive stats:', err);
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
  if (btn) {
    btn.disabled = true;
    btn.innerText = '正在全量归档...';
  }
  try {
    const res = await fetch('/api/archive_all', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force: false })
    });
    const data = await res.json();
    showToast(`全量归档完成: 新增 ${data.success || 0} 张, 跳过 ${data.skipped || 0} 张`, 'success');
    loadArchiveStats();
  } catch (err) {
    console.error('Batch archive error:', err);
    showToast('批量归档执行异常', 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = '⚡ 补齐全量历史归档';
    }
  }
}

// ================== 高清关键帧微距放大视窗 (Lightbox) ==================

const lightboxState = {
  isOpen: false,
  scale: 1.0,
  minScale: 0.5,
  maxScale: 6.0,
  translateX: 0,
  translateY: 0,
  isPanning: false,
  startX: 0,
  startY: 0,
  startTranslateX: 0,
  startTranslateY: 0,
  currentPos: '',
  currentTimestamp: 0
};

function initLightbox() {
  // 绑定三联画卡片放大按钮
  const bStart = document.getElementById('btn-zoom-start');
  if (bStart) bStart.addEventListener('click', () => openLightbox('start'));
  const bMid = document.getElementById('btn-zoom-mid');
  if (bMid) bMid.addEventListener('click', () => openLightbox('mid'));
  const bEnd = document.getElementById('btn-zoom-end');
  if (bEnd) bEnd.addEventListener('click', () => openLightbox('end'));

  // 绑定控制条
  const btnIn = document.getElementById('btn-lightbox-zoom-in');
  if (btnIn) btnIn.addEventListener('click', () => zoomLightbox(1.25));
  const btnOut = document.getElementById('btn-lightbox-zoom-out');
  if (btnOut) btnOut.addEventListener('click', () => zoomLightbox(0.8));
  const btnReset = document.getElementById('btn-lightbox-reset');
  if (btnReset) btnReset.addEventListener('click', resetLightboxZoom);
  const btnActual = document.getElementById('btn-lightbox-actual');
  if (btnActual) btnActual.addEventListener('click', setLightboxActualSize);
  const btnClose = document.getElementById('btn-lightbox-close');
  if (btnClose) btnClose.addEventListener('click', closeLightbox);

  // 鼠标滚轮缩放与按住平移
  const viewport = document.getElementById('lightbox-viewport');
  const canvas = document.getElementById('lightbox-canvas');

  if (viewport && canvas) {
    viewport.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const mouseX = e.clientX - rect.left - rect.width / 2;
      const mouseY = e.clientY - rect.top - rect.height / 2;
      const factor = e.deltaY > 0 ? 0.85 : 1.18;
      zoomLightboxAt(factor, mouseX, mouseY);
    }, { passive: false });

    canvas.addEventListener('mousedown', (e) => {
      lightboxState.isPanning = true;
      lightboxState.startX = e.clientX;
      lightboxState.startY = e.clientY;
      lightboxState.startTranslateX = lightboxState.translateX;
      lightboxState.startTranslateY = lightboxState.translateY;
      canvas.classList.add('panning');
    });

    window.addEventListener('mousemove', (e) => {
      if (!lightboxState.isPanning) return;
      const dx = e.clientX - lightboxState.startX;
      const dy = e.clientY - lightboxState.startY;
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

  const range = getSegmentLocalRange(seg);
  const relTime = pos === 'start' ? range.start : (pos === 'end' ? range.end : (range.start + range.end) / 2.0);
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
  if (pos === 'mid' && cardImg && cardImg.src && (cardImg.src.includes('yolo') || cardImg.src.includes('data:image') || cardImg.src.includes('annotated'))) {
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
