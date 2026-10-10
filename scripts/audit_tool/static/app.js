// HomeVlog Streamlined Reviewer • 极速轻量打标核心交互引擎

const state = {
  currentCategory: 'gold_candidates', // 'gold_candidates' | 'all' | 'reviewed'
  searchQuery: '',
  hideReviewed: false,
  items: [],
  filteredItems: [],
  selectedIndex: -1,
  counts: { gold_candidates: 0, all: 0, reviewed: 0 },
  fourKeysMode: false,
};

const prefetchSet = new Set();
const frameCache = new Map();

document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  loadQueue();
});

// ================== 1. 事件监听与键盘绑定 ==================
function initEventListeners() {
  // 分类胶囊切换
  document.querySelectorAll('.filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const cat = chip.getAttribute('data-category');
      switchCategory(cat);
    });
  });

  // 搜索框
  const searchInput = document.getElementById('queue-search-input');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      state.searchQuery = e.target.value.toLowerCase().trim();
      applyFilterAndRender();
    });
  }

  // 隐藏已复核开关
  const toggleHide = document.getElementById('toggle-hide-reviewed');
  if (toggleHide) {
    toggleHide.addEventListener('change', (e) => {
      state.hideReviewed = e.target.checked;
      applyFilterAndRender();
    });
  }

  // 底部两键打标
  const btnKeep = document.getElementById('btn-action-keep');
  if (btnKeep) {
    btnKeep.addEventListener('click', (e) => handleDecision('keep', e.shiftKey));
  }
  const btnDrop = document.getElementById('btn-action-drop');
  if (btnDrop) {
    btnDrop.addEventListener('click', (e) => handleDecision('drop', e.shiftKey));
  }

  // 上下导航与撤销
  const btnPrev = document.getElementById('btn-prev');
  if (btnPrev) btnPrev.addEventListener('click', () => selectNext(-1));
  const btnNext = document.getElementById('btn-next');
  if (btnNext) btnNext.addEventListener('click', () => selectNext(1));
  const btnClear = document.getElementById('btn-clear');
  if (btnClear) btnClear.addEventListener('click', clearCurrentReview);

  // 四键模式开关 (原位平滑内嵌切换，两键与四键互斥，杜绝向上遮挡)
  const btnToggle4 = document.getElementById('btn-toggle-4keys');
  const drawer4 = document.getElementById('quad-keys-drawer');
  const twoKeysGroup = document.getElementById('two-keys-group');
  if (btnToggle4 && drawer4) {
    btnToggle4.addEventListener('click', () => {
      state.fourKeysMode = !state.fourKeysMode;
      drawer4.classList.toggle('hidden', !state.fourKeysMode);
      if (twoKeysGroup) {
        twoKeysGroup.classList.toggle('hidden', state.fourKeysMode);
      }
      btnToggle4.innerText = state.fourKeysMode ? '收起四键 ▴' : '四键模式 ▾';
    });
  }

  // 四键独立按钮
  document.querySelectorAll('.btn-quad').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const label = btn.getAttribute('data-label');
      submitReviewDirect(label, e.shiftKey);
    });
  });

  // 工具下拉菜单
  const btnMenu = document.getElementById('btn-tools-menu');
  const menuPanel = document.getElementById('tools-menu-panel');
  if (btnMenu && menuPanel) {
    btnMenu.addEventListener('click', (e) => {
      e.stopPropagation();
      menuPanel.classList.toggle('hidden');
    });
    document.addEventListener('click', () => {
      menuPanel.classList.add('hidden');
    });
  }

  // 导出黄金基准
  const btnExportGold = document.getElementById('btn-export-gold');
  if (btnExportGold) {
    btnExportGold.addEventListener('click', exportGoldBenchmark);
  }

  // 调参仿真 (What-If) 模态框
  const btnOpenWhatif = document.getElementById('btn-open-whatif');
  const modalWhatif = document.getElementById('modal-whatif');
  const btnCloseWhatif = document.getElementById('btn-close-whatif');
  if (btnOpenWhatif && modalWhatif) {
    btnOpenWhatif.addEventListener('click', () => modalWhatif.classList.remove('hidden'));
  }
  if (btnCloseWhatif && modalWhatif) {
    btnCloseWhatif.addEventListener('click', () => modalWhatif.classList.add('hidden'));
  }

  const btnRunSim = document.getElementById('btn-run-simulation');
  if (btnRunSim) {
    btnRunSim.addEventListener('click', runTuningSimulation);
  }

  const btnApplyCfg = document.getElementById('btn-apply-tuning-config');
  if (btnApplyCfg) {
    btnApplyCfg.addEventListener('click', applyTuningConfig);
  }

  // 仿真输入回车监听
  ['sim-input-energy', 'sim-input-dur', 'sim-input-yolo'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          runTuningSimulation();
        }
      });
    }
  });

  // 视频播放模态框
  const btnClosePlayer = document.getElementById('btn-close-player');
  const modalPlayer = document.getElementById('modal-player');
  const videoEl = document.getElementById('vlog-video-el');
  if (btnClosePlayer && modalPlayer) {
    btnClosePlayer.addEventListener('click', () => {
      modalPlayer.classList.add('hidden');
      if (videoEl) videoEl.pause();
    });
  }

  // 模态框遮罩点击关闭 (惯性交互)
  [modalWhatif, modalPlayer].forEach(modal => {
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) {
          modal.classList.add('hidden');
          if (modal === modalPlayer && videoEl) {
            videoEl.pause();
          }
        }
      });
    }
  });

  // 重新浓缩成片
  const btnRerender = document.getElementById('btn-open-rerender');
  if (btnRerender) {
    btnRerender.addEventListener('click', triggerRerender);
  }

  // 全局键盘盲打
  window.addEventListener('keydown', handleKeydown);
}

function handleKeydown(e) {
  // ESC 键优先关闭打开的模态框
  if (e.key === 'Escape') {
    const modalWhatif = document.getElementById('modal-whatif');
    if (modalWhatif && !modalWhatif.classList.contains('hidden')) {
      e.preventDefault();
      modalWhatif.classList.add('hidden');
      return;
    }
    const modalPlayer = document.getElementById('modal-player');
    if (modalPlayer && !modalPlayer.classList.contains('hidden')) {
      e.preventDefault();
      modalPlayer.classList.add('hidden');
      const videoEl = document.getElementById('vlog-video-el');
      if (videoEl) videoEl.pause();
      return;
    }
  }

  // 如果在输入框内打字，不拦截按键
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  const isShift = e.shiftKey;

  if (e.key === '1' || e.code === 'Digit1') {
    e.preventDefault();
    if (state.fourKeysMode) {
      submitReviewDirect('CONFIRMED_MOTION', isShift);
    } else {
      handleDecision('keep', isShift);
    }
  } else if (e.key === '2' || e.code === 'Digit2') {
    e.preventDefault();
    if (state.fourKeysMode) {
      submitReviewDirect('FALSE_ALARM', isShift);
    } else {
      handleDecision('drop', isShift);
    }
  } else if (e.key === '3' || e.code === 'Digit3') {
    e.preventDefault();
    submitReviewDirect('MISSED_MOTION', isShift);
  } else if (e.key === '4' || e.code === 'Digit4') {
    e.preventDefault();
    submitReviewDirect('CONFIRMED_STATIC', isShift);
  } else if (e.key === 'j' || e.key === 'J' || e.key === 'ArrowDown') {
    e.preventDefault();
    selectNext(1);
  } else if (e.key === 'k' || e.key === 'K' || e.key === 'ArrowUp') {
    e.preventDefault();
    selectNext(-1);
  } else if (e.key === 'u' || e.key === 'U') {
    e.preventDefault();
    clearCurrentReview();
  }
}

// ================== 2. 数据拉取与队列渲染 ==================
async function loadQueue() {
  const container = document.getElementById('queue-list-container');
  if (container) {
    container.innerHTML = '<div style="padding: 20px; text-align: center; color: #64748b; font-size: 12px;">⏳ 正在调取高争议切片...</div>';
  }

  try {
    const res = await fetch(`/api/anomalies?category=${state.currentCategory}&limit=100`);
    const data = await res.json();
    state.items = Array.isArray(data) ? data : [];

    updateCategoryBadges();
    applyFilterAndRender();

    // 默认定位到第一个未复核的切片
    if (state.filteredItems.length > 0) {
      const firstUnreviewed = state.filteredItems.findIndex(it => !it.manual_label);
      selectItem(firstUnreviewed >= 0 ? firstUnreviewed : 0);
    } else {
      showEmptyPlaceholder();
    }
  } catch (err) {
    console.error('Load queue failed:', err);
    if (container) {
      container.innerHTML = '<div style="padding: 20px; text-align: center; color: #f43f5e; font-size: 12px;">⚠️ 加载切片队列失败</div>';
    }
    showEmptyPlaceholder();
  }
}

function showEmptyPlaceholder() {
  state.selectedIndex = -1;
  const placeholder = document.getElementById('stage-empty-placeholder');
  if (placeholder) placeholder.classList.remove('hidden');
  const playerImg = document.getElementById('main-player-img');
  if (playerImg) {
    playerImg.src = '';
    playerImg.classList.add('hidden');
  }
  const heroBadge = document.getElementById('hero-badge');
  if (heroBadge) heroBadge.classList.add('hidden');
  ['start', 'mid', 'end'].forEach(pos => {
    const im = document.getElementById(`img-thumb-${pos}`);
    if (im) im.removeAttribute('src');
  });
  setElementText('seg-id-badge', '#--');
  setElementText('seg-state-badge', 'STATE');
  setElementText('seg-time-range', '00:00:00 ~ 00:00:00');
  setElementText('seg-dur-badge', '0.0s');
  setElementText('seg-reason-badge', '--');
  const clusterTag = document.getElementById('seg-cluster-badge');
  if (clusterTag) clusterTag.classList.add('hidden');
}

function switchCategory(cat) {
  state.currentCategory = cat;
  document.querySelectorAll('.filter-chip').forEach(c => {
    c.classList.toggle('active', c.getAttribute('data-category')) === cat;
  });
  loadQueue();
}

function updateCategoryBadges() {
  const reviewed = state.items.filter(i => !!i.manual_label).length;
  const countGold = document.getElementById('count-gold');
  const countAll = document.getElementById('count-all');
  const countReviewed = document.getElementById('count-reviewed');

  if (state.currentCategory === 'gold_candidates' && countGold) {
    countGold.innerText = state.items.length;
  }
  if (countReviewed) {
    countReviewed.innerText = reviewed;
  }
  updateBatchProgressBar();
}

function updateBatchProgressBar() {
  const total = state.filteredItems.length;
  const reviewed = state.filteredItems.filter(i => !!i.manual_label).length;
  const textEl = document.getElementById('batch-progress-text');
  const fillEl = document.getElementById('batch-progress-fill');
  if (textEl) textEl.innerText = `${reviewed} / ${total}`;
  if (fillEl) {
    const pct = total > 0 ? (reviewed / total) * 100 : 0;
    fillEl.style.width = `${pct}%`;
  }
}

function applyFilterAndRender() {
  let list = [...state.items];
  if (state.hideReviewed) {
    list = list.filter(it => !it.manual_label);
  }
  if (state.searchQuery) {
    const q = state.searchQuery;
    list = list.filter(it => {
      const fn = (it.filepath || '').toLowerCase();
      const r = (it.reason_desc || '').toLowerCase();
      const s = (it.state || '').toLowerCase();
      return fn.includes(q) || r.includes(q) || s.includes(q) || String(it.id).includes(q);
    });
  }
  state.filteredItems = list;
  renderQueueList();
  updateBatchProgressBar();
  if (state.filteredItems.length === 0) {
    showEmptyPlaceholder();
  }
}

function renderQueueList() {
  const container = document.getElementById('queue-list-container');
  if (!container) return;
  container.innerHTML = '';

  if (state.filteredItems.length === 0) {
    container.innerHTML = '<div style="padding: 24px; text-align: center; color: #64748b; font-size: 12px;">队列已清空，全部复核完成 🎉</div>';
    return;
  }

  state.filteredItems.forEach((item, index) => {
    const el = document.createElement('div');
    el.className = `queue-item ${index === state.selectedIndex ? 'active' : ''}`;
    el.id = `queue-item-${item.id}`;

    // 状态图标
    const isDyn = ['DYNAMIC', 'PRESENCE', 'DYNAMIC_AUDIO'].includes(item.state);
    const icon = isDyn ? '⚡' : '⚪';

    // 打标徽章
    let badgeHtml = '';
    if (item.manual_label) {
      const m = item.manual_label;
      if (['CONFIRMED_MOTION', 'TP'].includes(m)) badgeHtml = '<span class="item-badge tp">TP</span>';
      else if (['FALSE_ALARM', 'FP'].includes(m)) badgeHtml = '<span class="item-badge fp">FP</span>';
      else if (['MISSED_MOTION', 'FN'].includes(m)) badgeHtml = '<span class="item-badge fn">FN</span>';
      else if (['CONFIRMED_STATIC', 'TN'].includes(m)) badgeHtml = '<span class="item-badge tn">TN</span>';
    } else if (item.cluster_count && item.cluster_count > 1) {
      badgeHtml = `<span class="item-badge cluster">x${item.cluster_count}</span>`;
    }

    const dur = (item.duration || 0).toFixed(1);
    const reason = item.reason_desc || item.state || '';

    el.innerHTML = `
      <div class="item-left">
        <span class="item-status-icon">${icon}</span>
        <div class="item-meta">
          <span class="item-id">#${item.id} <small style="color:#64748b;">${item.state}</small></span>
          <span class="item-reason" title="${reason}">${reason}</span>
        </div>
      </div>
      <div class="item-right">
        <span class="item-dur font-mono">${dur}s</span>
        ${badgeHtml}
      </div>
    `;

    el.addEventListener('click', () => selectItem(index));
    container.appendChild(el);
  });
}

// ================== 3. 切片选择与微动播放 ==================
function selectItem(index) {
  if (index < 0 || index >= state.filteredItems.length) return;
  state.selectedIndex = index;
  const item = state.filteredItems[index];

  // 更新左侧高亮
  document.querySelectorAll('.queue-item').forEach(el => el.classList.remove('active'));
  const currentCard = document.getElementById(`queue-item-${item.id}`);
  if (currentCard) {
    currentCard.classList.add('active');
    currentCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // 更新顶部胶囊信息
  setElementText('seg-id-badge', `#${item.id}`);
  setElementText('seg-state-badge', item.state);
  setElementText('seg-time-range', `${item.wall_start_str || ''} ~ ${item.wall_end_str || ''}`);
  setElementText('seg-dur-badge', `${(item.duration || 0).toFixed(1)}s (能量 ${(item.max_energy || 0).toFixed(1)})`);
  setElementText('seg-reason-badge', item.reason_desc || '候选样本');

  // 聚类提示
  const clusterTag = document.getElementById('seg-cluster-badge');
  const sweepHint = document.getElementById('cluster-sweep-hint');
  if (item.cluster_count && item.cluster_count > 1) {
    if (clusterTag) {
      clusterTag.classList.remove('hidden');
      clusterTag.innerText = `⚡ 关联聚类 x${item.cluster_count}`;
    }
    if (sweepHint) sweepHint.classList.remove('hidden');
  } else {
    if (clusterTag) clusterTag.classList.add('hidden');
    if (sweepHint) sweepHint.classList.add('hidden');
  }

  // 更新两键辅助文案
  const subKeep = document.getElementById('sub-keep-desc');
  const subDrop = document.getElementById('sub-drop-desc');
  const isDyn = ['DYNAMIC', 'PRESENCE', 'DYNAMIC_AUDIO'].includes(item.state);
  if (subKeep) subKeep.innerText = isDyn ? '确认原速 • TP' : '挽救微动 • FN';
  if (subDrop) subDrop.innerText = isDyn ? '纠偏光影 • FP' : '确认静止 • TN';

  // 加载视觉画面 (主图与微缩帧)
  loadVisualsForSegment(item);

  // 后台静默预加载相邻样本
  prefetchNeighbors(index);
}

function selectNext(direction) {
  const nextIdx = state.selectedIndex + direction;
  if (nextIdx >= 0 && nextIdx < state.filteredItems.length) {
    selectItem(nextIdx);
  }
}

function loadVisualsForSegment(item) {
  const placeholder = document.getElementById('stage-empty-placeholder');
  if (placeholder) placeholder.classList.add('hidden');

  const fp = item.filepath;
  const st = item.start_time;
  const et = item.end_time;
  const mid = (st + et) / 2.0;

  const playerImg = document.getElementById('main-player-img');
  const spinner = document.getElementById('player-spinner');
  const heroBadge = document.getElementById('hero-badge');

  if (heroBadge) {
    heroBadge.innerText = 'PEAK FRAME';
    heroBadge.classList.remove('hidden');
  }

  // 1. 优先加载峰值静态帧，确保 0ms 瞬间有画面
  const peakUrl = `/api/frame?filepath=${encodeURIComponent(fp)}&t=${mid.toFixed(3)}&w=800`;
  if (playerImg) {
    playerImg.src = peakUrl;
    playerImg.classList.remove('hidden');
  }

  // 2. 加载底部三联小卡片 (起点、峰值、终点)
  loadThumbImage('start', fp, st);
  loadThumbImage('mid', fp, mid);
  loadThumbImage('end', fp, Math.max(st, et - 0.1));

  // 3. 异步尝试载入 WebP 循环短微视频
  if (spinner) spinner.classList.remove('hidden');
  const clipUrl = `/api/clip?filepath=${encodeURIComponent(fp)}&start=${st.toFixed(2)}&end=${et.toFixed(2)}`;

  const clipLoader = new Image();
  clipLoader.onload = () => {
    // 仅在当前选中的切片依然是此切片时才替换
    if (state.selectedIndex >= 0 && state.filteredItems[state.selectedIndex]?.id === item.id) {
      if (playerImg) {
        playerImg.src = clipUrl;
        playerImg.classList.remove('hidden');
      }
      if (spinner) spinner.classList.add('hidden');
      if (heroBadge) {
        heroBadge.innerText = 'DYNAMIC LOOP (10fps)';
        heroBadge.classList.remove('hidden');
      }
    }
  };
  clipLoader.onerror = () => {
    if (spinner) spinner.classList.add('hidden');
  };
  clipLoader.src = clipUrl;
}

function loadThumbImage(pos, filepath, timestamp) {
  const img = document.getElementById(`img-thumb-${pos}`);
  const card = document.getElementById(`thumb-${pos}`);
  if (!img) return;

  const url = `/api/frame?filepath=${encodeURIComponent(filepath)}&t=${timestamp.toFixed(3)}&w=240`;
  img.src = url;

  // 鼠标悬停到小卡片时，主视窗立刻切换为该帧，提供精确视察
  if (card) {
    card.onmouseenter = () => {
      const playerImg = document.getElementById('main-player-img');
      const heroBadge = document.getElementById('hero-badge');
      if (playerImg) playerImg.src = url;
      if (heroBadge) heroBadge.innerText = `${pos.toUpperCase()} FRAME`;
    };
  }
}

// 环形预加载
function prefetchNeighbors(curIndex) {
  const neighbors = [curIndex + 1, curIndex + 2, curIndex - 1];
  neighbors.forEach(idx => {
    if (idx >= 0 && idx < state.filteredItems.length) {
      const target = state.filteredItems[idx];
      const mid = (target.start_time + target.end_time) / 2.0;
      const url = `/api/frame?filepath=${encodeURIComponent(target.filepath)}&t=${mid.toFixed(3)}&w=800`;
      if (!prefetchSet.has(url)) {
        prefetchSet.add(url);
        const im = new Image();
        im.src = url;
      }
    }
  });
}

// ================== 4. 裁决打标逻辑 (乐观更新 Optimistic UI) ==================
function handleDecision(action, isBatch = false) {
  if (state.selectedIndex < 0) return;
  const item = state.filteredItems[state.selectedIndex];
  const isDyn = ['DYNAMIC', 'PRESENCE', 'DYNAMIC_AUDIO'].includes(item.state);

  let manualLabel = '';
  if (action === 'keep') {
    manualLabel = isDyn ? 'CONFIRMED_MOTION' : 'MISSED_MOTION';
  } else if (action === 'drop') {
    manualLabel = isDyn ? 'FALSE_ALARM' : 'CONFIRMED_STATIC';
  }

  submitReviewDirect(manualLabel, isBatch);
}

async function submitReviewDirect(manualLabel, isBatch = false) {
  if (state.selectedIndex < 0) return;
  const item = state.filteredItems[state.selectedIndex];
  const doBatch = isBatch && Array.isArray(item.cluster_ids) && item.cluster_ids.length > 1;

  // 1. 乐观更新当前内存状态
  item.manual_label = manualLabel;
  if (doBatch) {
    const idSet = new Set(item.cluster_ids);
    state.items.forEach(it => {
      if (idSet.has(it.id)) it.manual_label = manualLabel;
    });
  }

  // 2. 局部刷新 UI (0ms 瞬时响应)
  updateCategoryBadges();
  renderQueueList();

  // 3. 自动步进到下一个未复核切片
  selectNext(1);

  // 4. 后台静默发送请求
  const endpoint = doBatch ? '/api/batch_review' : '/api/review';
  const payload = doBatch
    ? { segment_ids: item.cluster_ids, manual_label: manualLabel }
    : { segment_id: item.id, manual_label: manualLabel };

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      const hint = doBatch ? `已批量处理 ${data.count} 处切片` : `已标记 #${item.id}`;
      showToast(`${hint} ➔ ${manualLabel}`, 'success');
    }
  } catch (err) {
    console.error('Submit review error:', err);
    showToast('网络提交异常，请检查本地后台', 'error');
  }
}

async function clearCurrentReview() {
  if (state.selectedIndex < 0) return;
  const item = state.filteredItems[state.selectedIndex];
  if (!item.manual_label) {
    showToast('当前切片未打标，无需撤销', 'info');
    return;
  }

  item.manual_label = null;
  updateCategoryBadges();
  renderQueueList();

  try {
    await fetch('/api/clear_review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ segment_id: item.id })
    });
    showToast(`已撤销 #${item.id} 的复核标记`, 'info');
  } catch (err) {
    console.error('Clear review failed:', err);
  }
}

// ================== 5. 工具功能与导出 ==================
async function exportGoldBenchmark() {
  try {
    showToast('⏳ 正在导出黄金评测基准...', 'info');
    const res = await fetch('/api/export_gold_benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ version: '1.0', incremental_merge: true })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`🏆 成功导出 ${data.count} 条黄金基准至 tests/benchmark_data/`, 'success');
    } else {
      showToast('导出基准失败: ' + (data.error || '未知原因'), 'error');
    }
  } catch (err) {
    showToast('网络异常: ' + err.message, 'error');
  }
}

let lastSimulationParams = null;

async function runTuningSimulation() {
  const energyVal = parseFloat(document.getElementById('sim-input-energy').value) || 2.5;
  const durVal = parseFloat(document.getElementById('sim-input-dur').value) || 2.0;
  const yoloVal = parseFloat(document.getElementById('sim-input-yolo').value) || 0.3;

  const candidateParams = {
    'detection.min_motion_threshold': energyVal,
    'segment.min_motion_duration': durVal,
    'yolo.confidence': yoloVal,
  };
  lastSimulationParams = candidateParams;

  showToast('⚡ 正在对 100 样本黄金基准执行 What-If 推演...', 'info');
  try {
    const res = await fetch('/api/tuning_simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate_params: candidateParams })
    });
    const data = await res.json();
    if (data.error) {
      showToast('仿真失败: ' + data.error, 'error');
      return;
    }

    // 渲染报告
    const box = document.getElementById('sim-report-box');
    box.classList.remove('hidden');

    const vTag = document.getElementById('sim-verdict-tag');
    const vReason = document.getElementById('sim-verdict-reason');
    vTag.innerText = data.verdict;
    vReason.innerText = data.rationale;

    if (data.verdict === 'IMPROVED') {
      vTag.style.background = '#10b981';
    } else if (data.verdict === 'FAVORABLE_TRADEOFF') {
      vTag.style.background = '#3b82f6';
    } else if (data.verdict === 'REGRESSED') {
      vTag.style.background = '#ef4444';
    } else {
      vTag.style.background = '#64748b';
    }

    setElementText('stat-base-f1', `${data.baseline.f1}%`);
    setElementText('stat-sim-f1', `${data.simulated.f1}%`);
    const deltaSign = data.deltas.f1 >= 0 ? '+' : '';
    setElementText('stat-delta-f1', `${deltaSign}${data.deltas.f1}%`);
    setElementText('stat-fp-red', `-${data.deltas.fp_reduction}`);
    setElementText('stat-fn-inc', `+${data.deltas.fn_increase}`);

    // 显示应用配置按钮
    const btnApply = document.getElementById('btn-apply-tuning-config');
    btnApply.classList.remove('hidden');

    showToast(`推演完成: F1 预计变化 ${deltaSign}${data.deltas.f1}%`, 'success');
  } catch (err) {
    showToast('网络错误: ' + err.message, 'error');
  }
}

async function applyTuningConfig() {
  if (!lastSimulationParams) {
    showToast('未检测到仿真参数，请先执行推演', 'info');
    return;
  }
  if (!confirm('确定将当前仿真参数应用至生产配置文件 settings.yaml 吗？系统将自动创建备份文件。')) {
    return;
  }

  showToast('💾 正在安全写入配置并热重载...', 'info');
  try {
    const res = await fetch('/api/tuning_apply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params: lastSimulationParams })
    });
    const data = await res.json();
    if (data.success) {
      showToast('✅ 配置已安全更新并备份，热重载已生效！', 'success');
      const btnApply = document.getElementById('btn-apply-tuning-config');
      btnApply.classList.add('hidden');
    } else {
      showToast('写入配置失败: ' + (data.error || '校验不通过'), 'error');
    }
  } catch (err) {
    showToast('网络异常: ' + err.message, 'error');
  }
}

function openVideoPlayer(filepath, title = '') {
  const modalPlayer = document.getElementById('modal-player');
  const videoEl = document.getElementById('vlog-video-el');
  const titleEl = document.getElementById('player-modal-title');
  if (!modalPlayer || !videoEl) return;

  if (title && titleEl) {
    titleEl.innerText = `▶ ${title}`;
  }
  videoEl.src = `/api/video_stream?filepath=${encodeURIComponent(filepath)}`;
  modalPlayer.classList.remove('hidden');
  videoEl.play().catch(e => console.log('Autoplay blocked:', e));
}

async function triggerRerender() {
  const currentItem = state.selectedIndex >= 0 ? state.filteredItems[state.selectedIndex] : (state.items[0] || null);
  const defaultDate = currentItem?.date || new Date().toISOString().slice(0, 10).replace(/-/g, '');
  const defaultCam = currentItem?.cam_index ?? 0;

  const inputDate = prompt('请输入需要基于当前复核重新浓缩的日期 (YYYYMMDD):', defaultDate);
  if (!inputDate) return;

  showToast(`🚀 正在为 ${inputDate} (机位${defaultCam}) 启动后台重渲染...`, 'info');
  try {
    const res = await fetch('/api/rerender', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ date: inputDate, cam_index: defaultCam, version: 'reviewed' })
    });
    const data = await res.json();
    if (data.status === 'ALREADY_RUNNING' || data.status === 'CHECKING' || data.status === 'RENDERING') {
      pollRerenderProgress(inputDate, defaultCam);
    } else {
      showToast('启动重渲染失败: ' + (data.message || data.error || '未知错误'), 'error');
    }
  } catch (err) {
    showToast('网络请求失败: ' + err.message, 'error');
  }
}

function pollRerenderProgress(date, camIndex) {
  const timer = setInterval(async () => {
    try {
      const res = await fetch(`/api/rerender_status?date=${date}&cam_index=${camIndex}`);
      const st = await res.json();
      if (st.status === 'COMPLETED') {
        clearInterval(timer);
        showToast(`🎉 重渲染完成 (${st.file_size_mb} MB)，正在开启成片预览！`, 'success');
        if (st.output_file) {
          openVideoPlayer(st.output_file, `Daily Vlog 重渲染成片 (${date})`);
        }
      } else if (st.status === 'FAILED') {
        clearInterval(timer);
        showToast(`❌ 重渲染失败: ${st.error || '未知错误'}`, 'error');
      } else if (st.status === 'RENDERING' || st.status === 'CONCATING') {
        showToast(`⏳ 正在重渲染 (${st.progress}%: Batch ${st.current_batch}/${st.total_batches})`, 'info');
      }
    } catch (err) {
      clearInterval(timer);
    }
  }, 3000);
}

// 辅助函数
function setElementText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}

function showToast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const t = document.createElement('div');
  t.className = `toast-msg ${type}`;
  t.innerText = msg;
  container.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    setTimeout(() => t.remove(), 200);
  }, 1800);
}


