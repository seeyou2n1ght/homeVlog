// HomeVlog Studio Audit Console Client Logic

let state = {
  currentTab: 'anomalies', // 'anomalies' | 'tree'
  anomalies: [],
  filteredAnomalies: [],
  currentFile: null,
  currentSegments: [],
  selectedSegIndex: -1,
  sidebarCollapsed: false,
};

document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  loadOverview();
  loadAnomalies();
  loadFileTree();
  loadArchiveStats();
});

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

  // 搜索过滤
  const filterInput = document.getElementById('filter-input');
  if (filterInput) {
    filterInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      filterAnomalies(q);
    });
  }

  // 模式切换 Capsule
  document.getElementById('tab-anomalies').addEventListener('click', () => switchTab('anomalies'));
  document.getElementById('tab-tree').addEventListener('click', () => switchTab('tree'));

  // 报表导出
  document.getElementById('btn-export-csv').addEventListener('click', () => {
    showToast('正在生成并下载 CSV 报表...', 'info');
    window.location.href = '/api/export?format=csv';
  });
  document.getElementById('btn-export-json').addEventListener('click', () => {
    showToast('正在生成并下载 JSON 报表...', 'info');
    window.location.href = '/api/export?format=json';
  });

  // 工具按钮
  document.getElementById('btn-yolo-enhance').addEventListener('click', runYoloEnhance);
  document.getElementById('btn-clip-preview').addEventListener('click', showClipPreview);

  // 打标按钮
  document.querySelectorAll('.decision-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const label = btn.getAttribute('data-label');
      submitCurrentReview(label);
    });
  });

  // 真实反馈帧归档库弹窗事件
  const btnOpenArchive = document.getElementById('btn-open-archive-modal');
  if (btnOpenArchive) {
    btnOpenArchive.addEventListener('click', openArchiveModal);
  }
  const btnArchiveClose = document.getElementById('btn-archive-close');
  if (btnArchiveClose) btnArchiveClose.addEventListener('click', closeArchiveModal);
  const btnArchiveModalClose = document.getElementById('btn-archive-modal-close');
  if (btnArchiveModalClose) btnArchiveModalClose.addEventListener('click', closeArchiveModal);
  const btnArchiveBatchRun = document.getElementById('btn-archive-batch-run');
  if (btnArchiveBatchRun) btnArchiveBatchRun.addEventListener('click', triggerBatchArchive);

  // 重渲染弹窗事件
  const btnOpenRerender = document.getElementById('btn-open-rerender');
  if (btnOpenRerender) {
    btnOpenRerender.addEventListener('click', openRerenderModal);
  }
  document.getElementById('btn-rerender-close').addEventListener('click', closeRerenderModal);
  document.getElementById('btn-rerender-cancel').addEventListener('click', cancelOrCloseRerender);
  document.getElementById('btn-rerender-start').addEventListener('click', startRerender);
  document.getElementById('btn-rerender-open-dir').addEventListener('click', openRerenderOutputDir);

  // 模态框关闭
  document.getElementById('btn-modal-close').addEventListener('click', () => {
    document.getElementById('preview-modal').classList.add('hidden');
    document.getElementById('modal-preview-img').src = '';
  });

  // 全局快捷键
  window.addEventListener('keydown', handleKeydown);
}

function switchTab(tabName) {
  state.currentTab = tabName;
  const tabAno = document.getElementById('tab-anomalies');
  const tabTree = document.getElementById('tab-tree');
  const anoContent = document.getElementById('anomaly-content');
  const treeContent = document.getElementById('tree-content');
  const scopeDesc = document.getElementById('header-scope-desc');

  if (tabName === 'anomalies') {
    tabAno.classList.add('active');
    tabTree.classList.remove('active');
    anoContent.classList.remove('hidden');
    treeContent.classList.add('hidden');
    scopeDesc.innerText = '⚡ 疑难争议优先排查模式';
  } else {
    tabTree.classList.add('active');
    tabAno.classList.remove('active');
    treeContent.classList.remove('hidden');
    anoContent.classList.add('hidden');
    scopeDesc.innerText = '📁 全量素材文件浏览模式';
  }
}

function handleKeydown(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

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
  } else if (e.key === 'Escape') {
    document.getElementById('preview-modal').classList.add('hidden');
  }
}

function showToast(msg, type = 'success') {
  const container = document.getElementById('toast-container');
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

// 加载全局看板
async function loadOverview() {
  try {
    const res = await fetch('/api/overview');
    const data = await res.json();
    document.getElementById('metric-total-segs').innerText = `/ ${data.total_segments}`;
    document.getElementById('metric-reviewed-segs').childNodes[0].nodeValue = `${data.reviewed_segments} `;
    document.getElementById('metric-tp').innerText = data.labels.tp;
    document.getElementById('metric-fp').innerText = data.labels.fp;
    document.getElementById('metric-fn').innerText = data.labels.fn;
    document.getElementById('metric-prec-rec').innerText = `${data.metrics.precision}% / ${data.metrics.recall}%`;

    // 进度条百分比
    const pct = data.total_segments > 0 ? (data.reviewed_segments / data.total_segments) * 100 : 0;
    document.getElementById('metric-progress-bar').style.width = `${pct.toFixed(1)}%`;
  } catch (err) {
    console.error('Failed to load overview:', err);
  }
}

// 加载疑难排查列表
async function loadAnomalies() {
  try {
    const res = await fetch('/api/anomalies?limit=60');
    const list = await res.json();
    state.anomalies = list || [];
    state.filteredAnomalies = state.anomalies;
    document.getElementById('anomaly-count').innerText = state.anomalies.length;
    renderAnomalyList();

    if (state.anomalies.length > 0) {
      const container = document.getElementById('anomaly-items');
      if (container.firstChild && container.firstChild.classList) {
        container.firstChild.click();
      }
    }
  } catch (err) {
    console.error('Failed to load anomalies:', err);
  }
}

function filterAnomalies(query) {
  if (!query) {
    state.filteredAnomalies = state.anomalies;
  } else {
    state.filteredAnomalies = state.anomalies.filter(item => {
      const fn = item.filepath.split('\\').pop().split('/').pop().toLowerCase();
      const eng = item.max_energy.toString();
      return fn.includes(query) || eng.includes(query) || item.state.toLowerCase().includes(query);
    });
  }
  renderAnomalyList();
}

function renderAnomalyList() {
  const container = document.getElementById('anomaly-items');
  container.innerHTML = '';

  if (!state.filteredAnomalies || state.filteredAnomalies.length === 0) {
    container.innerHTML = '<div class="state-loader">暂无符合条件的争议样本</div>';
    return;
  }

  state.filteredAnomalies.forEach(item => {
    const div = document.createElement('div');
    div.className = `anomaly-card state-${item.state.toLowerCase()}`;
    div.id = `anomaly-item-${item.id}`;

    let reason = '边缘微动临界切片';
    let icon = '🔍';
    if (item.state === 'DYNAMIC' && item.avg_confidence === 0) {
      reason = '疑似光影假阳性 (无目标置信度)';
      icon = '⚠️';
    } else if (item.state === 'STATIC' && item.max_energy >= 1.5) {
      reason = '疑似微动作漏判 (能量临界)';
      icon = '🎯';
    } else if (item.duration < 3.0) {
      reason = '破碎跳变切片';
      icon = '⏱️';
    }

    const fn = item.filepath.split('\\').pop().split('/').pop();
    div.innerHTML = `
      <div class="anomaly-card-top">
        <span class="anomaly-id">#${item.id} • ${item.state}</span>
        <span class="anomaly-dur">${item.duration.toFixed(1)}s</span>
      </div>
      <div class="anomaly-reason-tag">
        <span>${icon}</span>
        <span>${reason}</span>
      </div>
      <div class="anomaly-card-footer font-mono">
        <span>⚡ 峰值: ${item.max_energy.toFixed(1)}</span>
        <span style="max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${fn}</span>
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

// 加载全量目录树
async function loadFileTree() {
  try {
    const res = await fetch('/api/tree');
    const tree = await res.json();
    const container = document.getElementById('tree-items');
    container.innerHTML = '';

    if (!tree || tree.length === 0) {
      container.innerHTML = '<div class="state-loader">暂无素材文件数据</div>';
      return;
    }

    tree.forEach(dayGroup => {
      const dayDiv = document.createElement('div');
      dayDiv.className = 'tree-date-group';
      dayDiv.innerHTML = `<div class="tree-date-header" style="font-size: 12px; font-weight: 700; color: #94a3b8; padding: 4px 8px;">📅 日期: ${dayGroup.date}</div>`;

      dayGroup.cameras.forEach(cam => {
        const camDiv = document.createElement('div');
        camDiv.className = 'tree-cam-group';
        camDiv.style.marginLeft = '8px';
        camDiv.innerHTML = `<div class="tree-cam-header" style="font-size: 11px; color: #64748b; padding: 2px 8px;">📹 摄像头 ${cam.cam_index} (${cam.files.length} 个视频)</div>`;

        cam.files.forEach(file => {
          const fDiv = document.createElement('div');
          fDiv.className = 'tree-file-item';
          fDiv.id = `tree-file-${file.id}`;
          fDiv.style.cssText = 'padding: 6px 10px; border-radius: 6px; font-size: 12px; cursor: pointer; display: flex; justify-content: space-between; margin-left: 8px; margin-bottom: 2px;';
          fDiv.innerHTML = `
            <span class="font-mono" style="max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${file.filename}</span>
            <span style="font-size: 11px; color: var(--text-muted);">${file.reviewed_count}/${file.seg_count}</span>
          `;
          fDiv.addEventListener('click', () => {
            document.querySelectorAll('.tree-file-item').forEach(el => el.style.background = 'transparent');
            fDiv.style.background = 'rgba(6, 182, 212, 0.15)';
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

// 加载单文件的所有分段
async function loadFileSegments(fileId, filepath, targetSegId = null) {
  try {
    const url = fileId ? `/api/file_segments?file_id=${fileId}` : `/api/file_segments?filepath=${encodeURIComponent(filepath)}`;
    const res = await fetch(url);
    const data = await res.json();

    if (!data.file) return;

    state.currentFile = data.file;
    state.currentSegments = data.segments || [];

    const fn = state.currentFile.filepath.split('\\').pop().split('/').pop();
    document.getElementById('current-filename').innerText = fn;
    document.getElementById('current-status-chip').innerText = 
      `PRE: ${state.currentFile.prescreen_status} • ANA: ${state.currentFile.analysis_status}`;

    renderTimelineBar();

    let targetIndex = 0;
    if (targetSegId) {
      const found = state.currentSegments.findIndex(s => s.id === targetSegId);
      if (found >= 0) targetIndex = found;
    }
    selectSegment(targetIndex);
  } catch (err) {
    console.error('Failed to load file segments:', err);
  }
}

// 渲染现代化时间轴
function renderTimelineBar() {
  const bar = document.getElementById('timeline-bar');
  bar.innerHTML = '';
  const segs = state.currentSegments;
  if (!segs || segs.length === 0) return;

  const totalDur = Math.max(state.currentFile.file_duration || 1, segs[segs.length - 1].end_time);

  // 刻度尺文本
  document.getElementById('ruler-start').innerText = '00:00.0';
  document.getElementById('ruler-mid').innerText = formatTime(totalDur / 2);
  document.getElementById('ruler-end').innerText = formatTime(totalDur);

  segs.forEach((s, idx) => {
    const block = document.createElement('div');
    block.className = 'seg-block';
    block.id = `seg-block-${idx}`;

    if (s.manual_label === 'FALSE_ALARM') {
      block.classList.add('state-fp');
    } else if (s.manual_label === 'MISSED_MOTION') {
      block.classList.add('state-fn');
    } else if (s.state === 'DYNAMIC') {
      block.classList.add('state-dynamic');
    } else {
      block.classList.add('state-static');
    }

    const pct = ((s.end_time - s.start_time) / totalDur) * 100;
    block.style.width = `${Math.max(pct, 0.4)}%`;

    block.addEventListener('click', () => selectSegment(idx));
    block.addEventListener('mouseenter', () => {
      document.getElementById('timeline-hover-info').innerText = 
        `切片 #${s.id} | ${formatTime(s.start_time)} ~ ${formatTime(s.end_time)} (${s.duration.toFixed(1)}s) | 状态: ${s.state} | 能量: ${s.max_energy.toFixed(1)}`;
    });

    bar.appendChild(block);
  });
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return `${m.toString().padStart(2, '0')}:${s.padStart(4, '0')}`;
}

// 选中某个切片
function selectSegment(index) {
  if (index < 0 || index >= state.currentSegments.length) return;
  state.selectedSegIndex = index;
  const seg = state.currentSegments[index];

  // 时间轴高亮
  document.querySelectorAll('.seg-block').forEach(b => b.classList.remove('selected'));
  const targetBlock = document.getElementById(`seg-block-${index}`);
  if (targetBlock) targetBlock.classList.add('selected');

  // 顶部标签徽标
  const tagsWrap = document.getElementById('current-segment-tags');
  tagsWrap.innerHTML = '';

  const stateChip = document.createElement('span');
  stateChip.className = `chip ${seg.state === 'DYNAMIC' ? 'chip-dyn' : 'chip-sta'}`;
  stateChip.innerText = `● 原判定: ${seg.state}`;
  tagsWrap.appendChild(stateChip);

  const durChip = document.createElement('span');
  durChip.className = 'chip font-mono';
  durChip.innerText = `⏱️ ${seg.duration.toFixed(1)}s [${seg.start_time.toFixed(1)}s - ${seg.end_time.toFixed(1)}s]`;
  tagsWrap.appendChild(durChip);

  const energyChip = document.createElement('span');
  energyChip.className = 'chip font-mono';
  energyChip.innerText = `⚡ 能量: ${seg.max_energy.toFixed(2)}`;
  tagsWrap.appendChild(energyChip);

  if (seg.manual_label) {
    const revChip = document.createElement('span');
    revChip.className = 'chip chip-reviewed';
    revChip.innerText = `✓ 已复核: ${seg.manual_label}`;
    tagsWrap.appendChild(revChip);
  }

  // 重置 YOLO
  document.getElementById('yolo-tags').innerHTML = 
    '<span class="inspector-placeholder">点击「⚡ 运行 YOLO 强化画框」即刻进行多目标分析与边界框定位</span>';

  loadFramesForSegment(seg);
}

function loadFramesForSegment(seg) {
  const fp = state.currentFile.filepath;
  const tStart = seg.start_time;
  const tMid = (seg.start_time + seg.end_time) / 2.0;
  const tEnd = Math.max(seg.end_time - 0.1, seg.start_time);

  document.getElementById('ts-start').innerText = `${tStart.toFixed(2)}s`;
  document.getElementById('ts-mid').innerText = `${tMid.toFixed(2)}s`;
  document.getElementById('ts-end').innerText = `${tEnd.toFixed(2)}s`;

  setFrameImage('img-start', 'shimmer-start', fp, tStart);
  setFrameImage('img-mid', 'shimmer-mid', fp, tMid);
  setFrameImage('img-end', 'shimmer-end', fp, tEnd);
}

function setFrameImage(imgId, shimmerId, filepath, timestamp) {
  const img = document.getElementById(imgId);
  const shimmer = document.getElementById(shimmerId);
  shimmer.classList.remove('hidden');

  const src = `/api/frame?filepath=${encodeURIComponent(filepath)}&t=${timestamp.toFixed(3)}&w=640`;
  const tempImg = new Image();
  tempImg.onload = () => {
    img.src = src;
    shimmer.classList.add('hidden');
  };
  tempImg.onerror = () => {
    shimmer.classList.add('hidden');
  };
  tempImg.src = src;
}

// 运行 YOLO 强化画框
async function runYoloEnhance() {
  if (state.selectedSegIndex < 0) return;
  const seg = state.currentSegments[state.selectedSegIndex];
  const tMid = (seg.start_time + seg.end_time) / 2.0;

  const btn = document.getElementById('btn-yolo-enhance');
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
      document.getElementById('img-mid').src = data.image_url;
      const tagsWrap = document.getElementById('yolo-tags');
      tagsWrap.innerHTML = '';
      if (!data.objects || data.objects.length === 0) {
        tagsWrap.innerHTML = '<span class="inspector-placeholder">⚠️ 未检出明确主体目标（高度疑似为光影/树影/窗帘晃动假动作）</span>';
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
  } catch (err) {
    console.error('YOLO enhance failed:', err);
    showToast('YOLO 增强检测失败', 'error');
  } finally {
    btn.innerHTML = oldText;
    btn.disabled = false;
  }
}

// 短动图预览弹窗
function showClipPreview() {
  if (state.selectedSegIndex < 0) return;
  const seg = state.currentSegments[state.selectedSegIndex];
  const modal = document.getElementById('preview-modal');
  const img = document.getElementById('modal-preview-img');
  const loading = document.getElementById('modal-loading');

  modal.classList.remove('hidden');
  img.src = '';
  loading.style.display = 'flex';

  const clipUrl = `/api/clip?filepath=${encodeURIComponent(state.currentFile.filepath)}&start=${seg.start_time}&end=${seg.end_time}`;
  const tempImg = new Image();
  tempImg.onload = () => {
    img.src = clipUrl;
    loading.style.display = 'none';
  };
  tempImg.src = clipUrl;
}

// 提交审核打标
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
      renderTimelineBar();
      loadOverview();
      loadArchiveStats();
      const labelMap = {
        'CONFIRMED_MOTION': '动态确认',
        'FALSE_ALARM': '误报光影',
        'MISSED_MOTION': '漏检补录',
        'CONFIRMED_STATIC': '静止确认'
      };
      const labelName = labelMap[manualLabel] || manualLabel;
      showToast(`已标记为 [${labelName}] 并自动归档代表帧`, 'success');
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
    selectSegment(nextIdx);
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
  document.getElementById('rerender-target-desc').innerText = `${date} (cam${cam})`;
  document.getElementById('rerender-file-check-status').className = 'status-chip';
  document.getElementById('rerender-file-check-status').innerText = '就绪待命';
  document.getElementById('rerender-stage-text').innerText = '准备重新浓缩';
  document.getElementById('rerender-pct-text').innerText = '0%';
  document.getElementById('rerender-progress-fill').style.width = '0%';
  document.getElementById('rerender-progress-fill').style.background = 'linear-gradient(90deg, #06b6d4, #3b82f6)';
  document.getElementById('rerender-batch-text').innerText = '批次: 0 / 0';
  document.getElementById('rerender-time-text').innerText = '已耗时: 0.0s';
  document.getElementById('rerender-missing-alert').classList.add('hidden');
  document.getElementById('rerender-missing-list').innerHTML = '';
  document.getElementById('btn-rerender-start').classList.remove('hidden');
  document.getElementById('btn-rerender-start').disabled = false;
  document.getElementById('btn-rerender-open-dir').classList.add('hidden');
  document.getElementById('btn-rerender-cancel').innerText = '关闭';

  document.getElementById('rerender-modal').classList.remove('hidden');

  // 查询当前是否有已在跑的该任务
  checkCurrentRerenderStatus(date, cam);
}

function closeRerenderModal() {
  if (rerenderPollTimer) {
    clearInterval(rerenderPollTimer);
    rerenderPollTimer = null;
  }
  document.getElementById('rerender-modal').classList.add('hidden');
}

function cancelOrCloseRerender() {
  const { date, cam } = getActiveDateAndCam();
  const btn = document.getElementById('btn-rerender-cancel');
  if (btn.innerText === '中断任务') {
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
  btnStart.disabled = true;
  document.getElementById('btn-rerender-cancel').innerText = '中断任务';

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
      btnStart.disabled = false;
      document.getElementById('btn-rerender-cancel').innerText = '关闭';
      return;
    }
    showToast(`已提交 ${date} cam${cam} 重浓缩任务`, 'info');
    startPollingRerender(date, cam);
  } catch (err) {
    console.error('Start rerender failed:', err);
    showToast('提交重浓缩请求异常', 'error');
    btnStart.disabled = false;
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
        document.getElementById('btn-rerender-cancel').innerText = '关闭';
        document.getElementById('btn-rerender-start').disabled = false;
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

  timeText.innerText = `已耗时: ${(st.elapsed_s || 0).toFixed(1)}s`;
  pctText.innerText = `${st.progress || 0}%`;
  barFill.style.width = `${st.progress || 0}%`;

  if (st.missing_files && st.missing_files.length > 0) {
    missingAlert.classList.remove('hidden');
    missingList.innerHTML = st.missing_files.map(f => `<div>• ${f.split('\\').pop().split('/').pop()}</div>`).join('');
    checkChip.className = 'status-chip';
    checkChip.style.background = 'rgba(245, 158, 11, 0.2)';
    checkChip.style.color = '#f59e0b';
    checkChip.innerText = `⚠️ 缺失 ${st.missing_files.length} 个物理文件 (已自愈跳过)`;
  } else if (st.status !== 'IDLE') {
    checkChip.className = 'status-chip';
    checkChip.style.background = 'rgba(16, 185, 129, 0.2)';
    checkChip.style.color = '#34d399';
    checkChip.innerText = '✅ 全部源文件物理校验通过';
  }

  if (st.status === 'CHECKING') {
    stageText.innerText = '正在探测素材物理存在性与可读性...';
  } else if (st.status === 'RENDERING') {
    stageText.innerText = `正在执行硬件加速切片压制 (批次 ${st.current_batch}/${st.total_batches})...`;
    batchText.innerText = `批次: ${st.current_batch} / ${st.total_batches}`;
  } else if (st.status === 'CONCATING') {
    stageText.innerText = '正在合并最终成片并封装元数据...';
  } else if (st.status === 'COMPLETED') {
    stageText.innerText = `🎉 浓缩压制完成！成片体积: ${st.file_size_mb || 0} MB`;
    barFill.style.background = '#10b981';
    btnStart.classList.add('hidden');
    btnOpenDir.classList.remove('hidden');
    currentRerenderOutputFile = st.output_file;
    showToast('Vlog 成片重新浓缩成功！', 'success');
  } else if (st.status === 'FAILED') {
    stageText.innerText = `❌ 重渲染失败: ${st.error || '未知错误'}`;
    barFill.style.background = '#f43f5e';
    showToast(`重渲染失败: ${st.error}`, 'error');
  } else if (st.status === 'CANCELLED') {
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
          'CONFIRMED_STATIC': { text: '静止确认', bg: 'rgba(148, 163, 184, 0.2)', color: '#94a3b8' },
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
  document.getElementById('archive-modal').classList.remove('hidden');
}

function closeArchiveModal() {
  document.getElementById('archive-modal').classList.add('hidden');
}

async function triggerBatchArchive() {
  const btn = document.getElementById('btn-archive-batch-run');
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

