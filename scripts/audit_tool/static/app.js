// HomeVlog Streamlined Reviewer • 极速轻量打标核心交互引擎

const state = {
  currentCategory: 'gold_candidates', // 'gold_candidates' | 'all' | 'reviewed'
  searchQuery: '',
  hideReviewed: false,
  items: [],
  filteredItems: [],
  selectedIndex: -1,
  counts: { gold_candidates: 0, all: 0, reviewed: 0 },
  offset: 0, pageSize: 50, totalItems: 0, seed: 0, requestSerial: 0, loading: false,
  scope: {}, visualReference: null, previewPinnedId: null,
  fourKeysMode: false,
};

let queueSearchTimer;
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
      clearTimeout(queueSearchTimer);
      queueSearchTimer = setTimeout(() => { state.offset = 0; loadQueue(); }, 300);
    });
  }

  // 隐藏已复核开关
  const toggleHide = document.getElementById('toggle-hide-reviewed');
  if (toggleHide) {
    toggleHide.addEventListener('change', (e) => {
      state.hideReviewed = e.target.checked;
      state.offset = 0; loadQueue();
    });
  }

  document.getElementById('queue-apply-scope')?.addEventListener('click', () => {
    const seed = Number(document.getElementById('queue-seed').value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) return showToast('种子须为0–2147483647整数', 'error');
    state.seed = seed;
    state.scope = Object.fromEntries(['date','camera','state','scenario'].map(key => [key, document.getElementById(`queue-${key}`).value.trim()]));
    state.offset = 0; loadQueue();
  });
  document.getElementById('queue-seed')?.addEventListener('change', e => {
    const seed = Number(e.target.value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) return showToast('种子须为0–2147483647整数', 'error');
    state.seed = seed; state.offset = 0; loadQueue();
  });
  document.getElementById('queue-resample')?.addEventListener('click', () => {
    state.seed = (state.seed + 1) % 2147483648;
    document.getElementById('queue-seed').value = state.seed;
    state.offset = 0; loadQueue();
  });
  document.getElementById('queue-page-prev')?.addEventListener('click', () => {
    if (!state.loading && state.offset > 0) { state.offset = Math.max(0, state.offset - state.pageSize); loadQueue(); }
  });
  document.getElementById('queue-page-next')?.addEventListener('click', () => {
    if (!state.loading && state.offset + state.pageSize < state.totalItems) { state.offset += state.pageSize; loadQueue(); }
  });

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

  document.getElementById('btn-index-visual-source')?.addEventListener('click', indexCurrentVisualSource);
  document.getElementById('btn-delivery-comparison')?.addEventListener('click', openDeliveryComparison);
  document.getElementById('btn-close-delivery-comparison')?.addEventListener('click', () => document.getElementById('modal-delivery-comparison').classList.add('hidden'));
  document.getElementById('btn-review-context')?.addEventListener('click', openReviewContext);
  document.getElementById('btn-close-context')?.addEventListener('click', () => document.getElementById('modal-context').classList.add('hidden'));
  document.getElementById('btn-save-review-context')?.addEventListener('click', saveReviewContext);
  // 视频播放模态框
  const btnClosePlayer = document.getElementById('btn-close-player');
  const modalPlayer = document.getElementById('modal-player');
  const videoEl = document.getElementById('vlog-video-el');
  if (btnClosePlayer && modalPlayer) {
    btnClosePlayer.addEventListener('click', closeVideoPlayer);
  }
  document.getElementById('btn-open-source')?.addEventListener('click', () => {
    const item = state.filteredItems[state.selectedIndex];
    if (!item) return showToast('请先选择审核片段', 'info');
    openVideoPlayer(item.filepath, `原片 #${item.id} · ${item.wall_start_str}–${item.wall_end_str}`, {
      start: Number(item.in_file_start), end: Number(item.in_file_end)
    });
  });
  document.getElementById('btn-replay-interval')?.addEventListener('click', () => {
    if (videoEl && playerInterval) {
      videoEl.currentTime = playerInterval.start;
      videoEl.play().catch(() => {});
    }
  });

  // 模态框遮罩点击关闭 (惯性交互)
  [modalPlayer].forEach(modal => {
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) {
          modal.classList.add('hidden');
          if (modal === modalPlayer) closeVideoPlayer();
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
    const contextModal = document.getElementById('modal-context');
    if (contextModal && !contextModal.classList.contains('hidden')) {
      e.preventDefault();
      contextModal.classList.add('hidden');
      return;
    }
    const renderModal = document.getElementById('modal-rerender');
    if (renderModal && !renderModal.classList.contains('hidden')) {
      e.preventDefault();
      renderModal.classList.add('hidden');
      return;
    }
    const modalPlayer = document.getElementById('modal-player');
    if (modalPlayer && !modalPlayer.classList.contains('hidden')) {
      e.preventDefault();
      closeVideoPlayer();
      return;
    }
    const comparison = document.getElementById('modal-delivery-comparison');
    if (comparison && !comparison.classList.contains('hidden')) { comparison.classList.add('hidden'); e.preventDefault(); return; }
  }

  // 如果在输入框内打字，不拦截按键
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
  const contextModal = document.getElementById('modal-context');
  if (contextModal && !contextModal.classList.contains('hidden')) return;
  const renderModal = document.getElementById('modal-rerender');
  if (renderModal && !renderModal.classList.contains('hidden')) return;
  const comparison = document.getElementById('modal-delivery-comparison');
  if (comparison && !comparison.classList.contains('hidden')) return;
  const playerOpen = !document.getElementById('modal-player')?.classList.contains('hidden');
  if (playerOpen && !['j', 'J', 'k', 'K', 'ArrowDown', 'ArrowUp'].includes(e.key)) return;

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
async function loadQueue(preferredId = null) {
  if (queueSearchTimer) clearTimeout(queueSearchTimer);
  queueSearchTimer = null;
  const serial = ++state.requestSerial;
  state.loading = true;
  state.items = []; state.filteredItems = []; showEmptyPlaceholder();
  const container = document.getElementById('queue-list-container');
  if (container) container.innerText = '正在读取当前范围...';
  updateQueuePage();
  const query = new URLSearchParams({category:state.currentCategory,limit:state.pageSize,offset:state.offset,
    seed:state.seed,search:state.searchQuery,hide_reviewed:state.hideReviewed ? '1' : '0'});
  if (state.currentCategory === 'visual') query.set('reference_id',state.visualReference);
  for (const [key,value] of Object.entries(state.scope)) if (value !== '') query.set(key === 'camera' ? 'cam_index' : key, value);
  try {
    const response = await fetch(`/api/review_queue?${query}`);
    const data = await response.json();
    if (serial !== state.requestSerial) return;
    if (!response.ok || data.error || !Array.isArray(data.items)) throw Error(data.error || '队列响应无效');
    state.counts = data.counts; state.totalItems = data.total_items;
    if (state.totalItems && state.offset >= state.totalItems) {
      state.offset = Math.floor((state.totalItems - 1) / state.pageSize) * state.pageSize;
      return loadQueue(preferredId);
    }
    state.items = data.items; prefetchSet.clear();
    state.filteredItems = [...state.items]; state.selectedIndex = -1;
    renderQueueList(); updateCategoryBadges();
    const preferred = state.filteredItems.findIndex(item => item.id === preferredId);
    if (state.filteredItems.length) selectItem(preferred >= 0 ? preferred : 0);
    setElementText('queue-scope-status', `当前范围：${data.counts.all}条，未审${data.counts.unreviewed}条。本队列${data.total_items}条/组。` +
      (state.currentCategory === 'gold_candidates' ? '同源5分钟/状态/场景代表；非视觉相似，仅标当前项。' :
       state.currentCategory === 'random' ? `种子${data.seed}；库存与范围不变时可复现，不能认证其余片段。` : '计数来自完整筛选范围。'));
    if (state.currentCategory === 'visual') setElementText('queue-scope-status', `参考#${state.visualReference} · 范围${data.counts.all}条，已采集索引${data.captured_index_items}条，当前匹配${data.indexed_items}条，本页有效${data.items.length}条。${data.status === 'UNAVAILABLE' ? '参考图缺少可信索引或源已变化；可从全部队列选择片段，再为当前原片建立索引。' : data.visual_notice}`);
    return true;
  } catch (error) {
    if (serial !== state.requestSerial) return;
    if (container) container.innerText = `加载失败：${error.message}`;
    setElementText('queue-scope-status', '库存读取失败；上次计数不能代表当前状态。');
    return false;
  } finally {
    if (serial === state.requestSerial) { state.loading = false; updateQueuePage(); }
  }
}

function updateQueuePage() {
  const pages = Math.max(1, Math.ceil(state.totalItems / state.pageSize));
  setElementText('queue-page-info', `${Math.floor(state.offset / state.pageSize)+1} / ${pages}`);
  const prev = document.getElementById('queue-page-prev'), next = document.getElementById('queue-page-next');
  if (prev) prev.disabled = state.loading || state.offset === 0;
  if (next) next.disabled = state.loading || state.offset + state.pageSize >= state.totalItems;
}

function showEmptyPlaceholder() {
  closeVideoPlayer();
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
  if (cat === 'visual') {
    const reference = state.filteredItems[state.selectedIndex];
    if (reference) state.visualReference = reference.id;
    if (!state.visualReference) return showToast('请先选择一个参考片段','info');
  }
  state.currentCategory = cat; state.offset = 0;
  if (cat === 'reviewed') { state.hideReviewed = false; document.getElementById('toggle-hide-reviewed').checked = false; }
  document.getElementById('queue-random-controls').classList.toggle('hidden', cat !== 'random');
  document.querySelectorAll('.filter-chip').forEach(c => {
    c.classList.toggle('active', c.getAttribute('data-category') === cat);
  });
  loadQueue();
}

function updateCategoryBadges() {
  setElementText('count-gold', state.counts.gold_candidates);
  setElementText('count-all', state.counts.all);
  setElementText('count-reviewed', state.counts.reviewed);
  updateBatchProgressBar();
}

function updateBatchProgressBar() {
  const total = state.filteredItems.length;
  const reviewed = state.filteredItems.filter(i => !!i.manual_label).length;
  const textEl = document.getElementById('batch-progress-text');
  const fillEl = document.getElementById('batch-progress-fill');
  if (textEl) textEl.innerText = `本页 ${reviewed} / ${total}`;
  if (fillEl) {
    const pct = total > 0 ? (reviewed / total) * 100 : 0;
    fillEl.style.width = `${pct}%`;
  }
}

function renderQueueList() {
  const container = document.getElementById('queue-list-container');
  if (!container) return;
  container.innerHTML = '';

  if (state.filteredItems.length === 0) {
    container.innerHTML = '<div style="padding: 24px; text-align: center; color: #64748b; font-size: 12px;">当前筛选无条目；不代表全部已复核</div>';
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
    } else if (item.represented_count > 1) {
      badgeHtml = `<span class="item-badge cluster">代表${Number(item.represented_count)}</span>`;
    } else if (item.cluster_count && item.cluster_count > 1) {
      badgeHtml = `<span class="item-badge cluster">x${item.cluster_count}</span>`;
    }

    const dur = (item.duration || 0).toFixed(1);
    const reason = escapeHTML((Number.isFinite(item.visual_distance) ? `视觉距离 ${item.visual_distance.toFixed(3)} · ` : '') + (item.reason_desc || item.state || ''));

    el.innerHTML = `
      <div class="item-left">
        <span class="item-status-icon">${icon}</span>
        <div class="item-meta">
          <span class="item-id">#${item.id} <small style="color:#64748b;">${escapeHTML(item.state)}</small></span>
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
  closeVideoPlayer();
  state.selectedIndex = index;
  state.previewPinnedId = null;
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
  const indexedFrame = Number.isFinite(item.visual_frame_time) ? item.visual_frame_time + Number(item.file_start_offset || 0) : null;
  const mid = indexedFrame ?? (st + et) / 2.0;
  setElementText('thumb-mid-label', indexedFrame == null ? '中点代表帧' : '索引代表帧');

  const playerImg = document.getElementById('main-player-img');
  const spinner = document.getElementById('player-spinner');
  const heroBadge = document.getElementById('hero-badge');

  if (heroBadge) {
    heroBadge.innerText = indexedFrame == null ? 'MIDPOINT FRAME · 代表图不等于整段' : '索引代表帧 · 相似画面不认证同一事件';
    heroBadge.classList.remove('hidden');
  }

  // 先显示代表帧，再加载短动图；用户选择帧后不被迟到动图替换。
  const representativeUrl = `/api/frame?filepath=${encodeURIComponent(fp)}&t=${mid.toFixed(3)}&w=800`;
  if (playerImg) {
    playerImg.onerror = () => {
      if (state.filteredItems[state.selectedIndex]?.id === item.id && heroBadge) {
        heroBadge.innerText = '画面提取失败 · 请检查原片';
      }
    };
    playerImg.src = representativeUrl;
    playerImg.classList.remove('hidden');
  }

  // 底部三联帧：起点、代表帧、终点。
  loadThumbImage('start', fp, st);
  loadThumbImage('mid', fp, mid);
  loadThumbImage('end', fp, Math.max(st, et - 0.1));

  // 3. 异步尝试载入 WebP 循环短微视频
  if (spinner) spinner.classList.remove('hidden');
  const clipUrl = `/api/clip?filepath=${encodeURIComponent(fp)}&start=${st.toFixed(2)}&end=${Math.min(et,st+6).toFixed(2)}`;

  const clipLoader = new Image();
  clipLoader.onload = () => {
    // 仅在当前选中的切片依然是此切片时才替换
    if (state.selectedIndex >= 0 && state.filteredItems[state.selectedIndex]?.id === item.id) {
      if (playerImg && state.previewPinnedId !== item.id) {
        playerImg.src = clipUrl;
        playerImg.classList.remove('hidden');
      }
      if (spinner) spinner.classList.add('hidden');
      if (heroBadge && state.previewPinnedId !== item.id) {
        heroBadge.innerText = '短预览 ≤6s · 10fps · 无声；完整复核请查看原片';
        heroBadge.classList.remove('hidden');
      }
    }
  };
  clipLoader.onerror = () => {
    if (spinner && state.filteredItems[state.selectedIndex]?.id === item.id) spinner.classList.add('hidden');
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
    const displayedId = state.filteredItems[state.selectedIndex]?.id;
    card.onmouseenter = () => {
      if (state.filteredItems[state.selectedIndex]?.id !== displayedId) return;
      state.previewPinnedId = displayedId;
      const playerImg = document.getElementById('main-player-img');
      const heroBadge = document.getElementById('hero-badge');
      if (playerImg) playerImg.src = url.replace('&w=240','&w=800');
      if (heroBadge) heroBadge.innerText = pos === 'mid' && Number.isFinite(state.filteredItems[state.selectedIndex]?.visual_frame_time) ? '索引代表帧 · 画面相似不认证同一事件' : `${pos.toUpperCase()} FRAME`;
    };
    card.onclick = card.onmouseenter;
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

const pendingReviewIds = new Set();

async function commitReview(ids, manualLabel, attribution = null) {
  if (ids.some(id => pendingReviewIds.has(id))) {
    showToast('该片段正在提交，请等待确认', 'info'); return false;
  }
  ids.forEach(id => pendingReviewIds.add(id));
  const batch = ids.length > 1;
  const payload = batch ? {segment_ids:ids, manual_label:manualLabel} : {segment_id:ids[0], manual_label:manualLabel};
  if (attribution) Object.assign(payload, attribution);
  try {
    const response = await fetch(batch ? '/api/batch_review' : '/api/review', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok || !data.success || (batch && data.count !== ids.length)) {
      showToast(`提交未完整确认：${data.error || '请刷新队列核对持久状态'}`, 'error');
      return false;
    }
    const targets = new Set(ids);
    state.items.forEach(item => { if (targets.has(item.id)) {
      item.manual_label = manualLabel;
      if (attribution) { item.scenario = attribution.scenario; item.user_notes = attribution.notes;
        if ('review_basis' in attribution) item.review_basis = attribution.review_basis; }
    } });
    updateCategoryBadges(); renderQueueList();
    showToast(`已确认保存${ids.length}条复核`, 'success');
    return true;
  } catch (err) {
    showToast(`提交结果未知：${err.message}；请刷新队列核对，未将界面标为成功。`, 'error');
    return false;
  } finally { ids.forEach(id => pendingReviewIds.delete(id)); }
}

async function submitReviewDirect(manualLabel, isBatch = false) {
  const item = state.filteredItems[state.selectedIndex];
  if (!item) return;
  const ids = isBatch && item.cluster_ids?.length > 1 ? item.cluster_ids : [item.id];
  const nextId = state.filteredItems[state.selectedIndex + 1]?.id;
  if (await commitReview(ids, manualLabel)) await loadQueue(nextId ?? item.id);
}

async function clearCurrentReview() {
  const item = state.filteredItems[state.selectedIndex];
  if (!item?.manual_label) return showToast('当前片段未打标', 'info');
  if (pendingReviewIds.has(item.id)) return;
  pendingReviewIds.add(item.id);
  try {
    const response = await fetch('/api/clear_review', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({segment_id:item.id})});
    const result = await response.json();
    if (!response.ok || !result.success) throw Error(result.error || '撤销未确认');
    item.manual_label = null; item.scenario = ''; item.user_notes = ''; item.review_basis = 'unknown';
    updateCategoryBadges(); renderQueueList();
    showToast(`已撤销 #${item.id}`, 'info');
    await loadQueue(item.id);
  } catch (err) { showToast(`撤销结果未知：${err.message}；请刷新核对。`, 'error'); }
  finally { pendingReviewIds.delete(item.id); }
}

let activeReviewContext = null;

async function fetchReviewContext(id) {
  const response = await fetch(`/api/review_context?segment_id=${id}`);
  const result = await response.json();
  if (!response.ok || result.error) throw Error(result.error || '配置/审核上下文不可用');
  return result;
}

function appendTableRow(body, values) {
  const row = document.createElement('tr');
  values.forEach(value => { const cell = document.createElement('td'); cell.innerText = value == null ? '—' : String(value); row.appendChild(cell); });
  body.appendChild(row);
}

async function openReviewContext() {
  const item = state.filteredItems[state.selectedIndex];
  if (!item) return showToast('请先选择审核片段', 'info');
  document.getElementById('modal-context').classList.remove('hidden');
  activeReviewContext = null;
  document.getElementById('btn-save-review-context').disabled = true;
  document.getElementById('review-context-parameters').replaceChildren();
  setElementText('review-context-status', '正在读取当前配置与审核计划...');
  try {
    const context = await fetchReviewContext(item.id);
    if (state.filteredItems[state.selectedIndex]?.id !== item.id) return;
    activeReviewContext = context;
    setElementText('review-context-title', `#${item.id} · 机位${context.segment.cam_index} · 配置与归因`);
    setElementText('review-context-status', context.notice);
    const label = document.getElementById('review-context-label');
    const normalized = {TP:'CONFIRMED_MOTION',VERIFIED_MOTION:'CONFIRMED_MOTION',FP:'FALSE_ALARM',FN:'MISSED_MOTION',TN:'CONFIRMED_STATIC'};
    label.value = normalized[context.segment.manual_label] || context.segment.manual_label || '';
    const scenario = document.getElementById('review-context-scenario');
    if (context.scenario && !Array.from(scenario.options).some(option => option.value === context.scenario)) {
      const option = document.createElement('option'); option.value = context.scenario; option.innerText = context.scenario; scenario.appendChild(option);
    }
    scenario.value = context.scenario || '';
    document.getElementById('review-context-notes').value = context.user_notes || '';
    document.getElementById('review-context-basis').value = context.review_basis || 'unknown';
    const analysis = context.comparison.analysis, reviewed = context.comparison.reviewed;
    const states = details => [...new Set(details.portions.map(portion => portion.state))].join(', ') || '无覆盖';
    setElementText('review-context-comparison', `保存的分析 → 人工覆盖计划：${states(analysis)} → ${states(reviewed)}；本区间展示贡献 ${analysis.display_s.toFixed(3)}s → ${reviewed.display_s.toFixed(3)}s。这是计划对比，成片效果需另行验证。`);
    const body = document.getElementById('review-context-parameters');
    context.parameters.forEach(parameter => appendTableRow(body, [parameter.param, parameter.global, parameter.camera, parameter.explicit_global ? '有显式值' : '预设/默认填充']));
    document.getElementById('btn-save-review-context').disabled = false;
  } catch (err) { setElementText('review-context-status', err.message); }
}

async function saveReviewContext() {
  if (!activeReviewContext) return;
  const context = activeReviewContext;
  const label = document.getElementById('review-context-label').value;
  if (!label) return showToast('待确认片段不提交裁决；确认后再保存。', 'info');
  const attribution = {scenario:document.getElementById('review-context-scenario').value,
    notes:document.getElementById('review-context-notes').value, review_basis:document.getElementById('review-context-basis').value};
  const button = document.getElementById('btn-save-review-context'); button.disabled = true;
  if (await commitReview([context.segment.id], label, attribution)) {
    const refreshed = await loadQueue(context.segment.id);
    setElementText('review-context-status', refreshed ? '保存已确认；队列库存已刷新。当前归因面板仍对应原片段。' : '保存已确认，但库存刷新失败，请手动刷新核对。');
    button.disabled = false;
  } else { button.disabled = false; }
}

let playerInterval = null;

function closeVideoPlayer() {
  const video = document.getElementById('vlog-video-el');
  document.getElementById('modal-player')?.classList.add('hidden');
  playerInterval = null;
  if (!video) return;
  video.onloadedmetadata = video.ontimeupdate = video.onerror = null;
  video.pause();
  video.removeAttribute('src');
  video.load();
}

function openVideoPlayer(filepath, title = '', interval = null, deliveryUrl = null) {
  const modalPlayer = document.getElementById('modal-player');
  const videoEl = document.getElementById('vlog-video-el');
  const titleEl = document.getElementById('player-modal-title');
  if (!modalPlayer || !videoEl) return;
  closeVideoPlayer();
  const hint = document.getElementById('player-interval-hint');
  const replay = document.getElementById('btn-replay-interval');
  if (interval && (!Number.isFinite(interval.start) || !Number.isFinite(interval.end)
      || interval.start < 0 || interval.end <= interval.start)) {
    showToast('审核区间无效，不能定位原片', 'error');
    return;
  }
  playerInterval = interval;
  replay?.classList.toggle('hidden', !interval);
  if (hint) hint.innerText = interval
    ? `原片文件内 ${interval.start.toFixed(2)}–${interval.end.toFixed(2)} 秒；播放完整审核区间，含原声。短动图最多6秒。`
    : '成片预览';

  if (title && titleEl) {
    titleEl.innerText = `▶ ${title}`;
  }
  videoEl.playbackRate = 1.0;
  videoEl.src = deliveryUrl || `/api/video_stream?filepath=${encodeURIComponent(filepath)}`;
  if (deliveryUrl && interval && hint) hint.innerText = `实际成片 ${interval.start.toFixed(3)}–${interval.end.toFixed(3)}秒；完整展示分段，含上下文及成片音轨。`;
  videoEl.onloadedmetadata = () => {
    if (interval && (!Number.isFinite(videoEl.duration) || interval.end > videoEl.duration + 0.05)) {
      videoEl.pause();
      playerInterval = null;
      replay?.classList.add('hidden');
      if (hint) hint.innerText = deliveryUrl ? '成片时长与交付映射不符，请刷新对比并核对交付。' : '原片时长不足以覆盖审核区间，请核对源文件与元数据。';
      return;
    }
    videoEl.currentTime = interval ? interval.start : 0;
    videoEl.play().catch(() => {
      if (hint) hint.innerText += ' 请点击播放。';
    });
  };
  videoEl.ontimeupdate = () => {
    if (playerInterval && videoEl.currentTime >= playerInterval.end) videoEl.pause();
  };
  videoEl.onerror = () => {
    if (hint) hint.innerText = deliveryUrl ? '成片版本已变化、读取失败或格式不受支持，请刷新对比后重试。' : '视频读取或浏览器解码失败，请检查共享连接与视频格式。';
  };
  modalPlayer.classList.remove('hidden');
  videoEl.load();
}

let reviewedRenderRequest = null;
let reviewedRenderTimer = null;
let reviewedRenderPolling = false;
let reviewedRenderLoading = false;

function showRenderBatchRows(reports, actual = false) {
  const body = document.getElementById('rerender-batches');
  body.replaceChildren();
  reports.forEach((entry) => {
    const report = actual ? entry.attempts?.at(-1) : entry;
    const row = document.createElement('tr');
    const values = [entry.batch_index + 1,
      report?.source_files?.map(fp => fp.split(/[\\/]/).at(-1)).join(', ') || '—',
      report?.expected_duration_s == null ? '—' : `${report.expected_duration_s.toFixed(2)}s`,
      actual ? '启动时快照' : (entry.review_intervals?.length || 0),
      actual ? `${entry.status} · ${entry.cache_hit == null ? '待判定' : (entry.cache_hit ? '已复用' : '重新渲染')}`
        : (entry.cache_hit ? '已校验可复用' : '需渲染（无匹配有效缓存）')];
    values.forEach(value => { const cell = document.createElement('td'); cell.innerText = value; row.appendChild(cell); });
    body.appendChild(row);
  });
}

async function triggerRerender() {
  if (reviewedRenderLoading || reviewedRenderTimer) {
    document.getElementById('modal-rerender').classList.remove('hidden');
    return;
  }
  const item = state.filteredItems[state.selectedIndex] || state.items[0];
  const modal = document.getElementById('modal-rerender');
  const dateInput = document.getElementById('rerender-date');
  const cameraInput = document.getElementById('rerender-camera');
  dateInput.value = item?.date || '';
  cameraInput.value = item?.cam_index ?? 0;
  modal.classList.remove('hidden');
  document.getElementById('btn-close-rerender').onclick = () => modal.classList.add('hidden');
  document.getElementById('btn-inspect-reviewed-render').onclick = inspectReviewedRender;
  [dateInput, cameraInput].forEach(input => { input.oninput = () => {
    reviewedRenderRequest = null;
    document.getElementById('btn-start-reviewed-render').disabled = true;
    setElementText('rerender-summary', '范围已改变，请刷新批次预览。');
  }; });
  await inspectReviewedRender();
}

async function inspectReviewedRender() {
  if (reviewedRenderLoading || reviewedRenderTimer) return;
  const date = document.getElementById('rerender-date').value.trim();
  const camIndex = Number(document.getElementById('rerender-camera').value);
  const modal = document.getElementById('modal-rerender');
  if (!/^\d{8}$/.test(date) || !Number.isInteger(camIndex) || camIndex < 0) {
    setElementText('rerender-summary', '请输入8位日期和非负整数机位。');
    return;
  }
  reviewedRenderLoading = true;
  const start = document.getElementById('btn-start-reviewed-render');
  const cancel = document.getElementById('btn-cancel-reviewed-render');
  modal.classList.remove('hidden');
  start.disabled = true;
  cancel.classList.add('hidden');
  reviewedRenderRequest = null;
  setElementText('rerender-summary', '正在读取审核快照与校验缓存；NAS读取可能需要时间。');
  document.getElementById('rerender-batches').replaceChildren();
  document.getElementById('btn-close-rerender').onclick = () => modal.classList.add('hidden');
  try {
    const response = await fetch(`/api/rerender_plan?date=${encodeURIComponent(date)}&cam_index=${camIndex}`);
    const plan = await response.json();
    if (date !== document.getElementById('rerender-date').value.trim() || camIndex !== Number(document.getElementById('rerender-camera').value)) return;
    if (plan.status !== 'AVAILABLE') {
      setElementText('rerender-summary', `预览不可用：${plan.error || ('不可访问素材 ' + (plan.missing_files?.length || 0))}`);
      return;
    }
    reviewedRenderRequest = {date, cam_index: camIndex, expected_snapshot: plan.snapshot_sha256};
    showRenderBatchRows(plan.batches);
    setElementText('rerender-summary', `${date} 机位${camIndex} · 共${plan.total_batches}批；已校验可复用${plan.reusable_batches}批、需渲染${plan.total_batches - plan.reusable_batches}批。`);
    start.disabled = false;
    start.onclick = startReviewedRender;
    cancel.onclick = async () => {
      if (!reviewedRenderRequest) return;
      try {
        const response = await fetch('/api/rerender_cancel', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(reviewedRenderRequest)});
        const data = await response.json();
        if (!data.success) showToast('任务已经结束或无法取消', 'info');
      } catch (err) { showToast(`取消请求失败：${err.message}`, 'error'); }
    };
  } catch (err) { setElementText('rerender-summary', `预览失败：${err.message}`); }
  finally { reviewedRenderLoading = false; }
}

async function startReviewedRender() {
  if (!reviewedRenderRequest) return;
  const request = {...reviewedRenderRequest};
  document.getElementById('btn-start-reviewed-render').disabled = true;
  setRenderRangeLocked(true);
  try {
    const response = await fetch('/api/rerender', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(request)});
    const data = await response.json();
    if (['ALREADY_RUNNING', 'CHECKING', 'RENDERING', 'CONCATING', 'CANCELLING'].includes(data.status)) {
      document.getElementById('btn-cancel-reviewed-render').classList.remove('hidden');
      pollRerenderProgress(request.date, request.cam_index);
    } else {
      setRenderRangeLocked(false);
      setElementText('rerender-summary', `启动失败：${data.message || data.error || '未知错误'}`);
    }
  } catch (err) {
    // A lost response does not prove the worker never started.
    pollRerenderProgress(request.date, request.cam_index);
    setElementText('rerender-summary', `启动响应未取得：${err.message}；正在核实任务状态。`);
  }
}

function setRenderRangeLocked(locked) {
  ['rerender-date', 'rerender-camera', 'btn-inspect-reviewed-render'].forEach(id => {
    document.getElementById(id).disabled = locked;
  });
}

function pollRerenderProgress(date, camIndex) {
  if (reviewedRenderTimer) clearInterval(reviewedRenderTimer);
  const poll = async () => {
    if (reviewedRenderPolling) return;
    reviewedRenderPolling = true;
    try {
      const response = await fetch(`/api/rerender_status?date=${date}&cam_index=${camIndex}`);
      const status = await response.json();
      showRenderBatchRows(status.batch_results || [], true);
      setElementText('rerender-summary', `${status.status} · ${status.progress || 0}% · 批次 ${status.current_batch || 0}/${status.total_batches || 0} · ${status.error || ''}`);
      if (['COMPLETED', 'FAILED', 'CANCELLED', 'IDLE'].includes(status.status)) {
        clearInterval(reviewedRenderTimer);
        reviewedRenderTimer = null;
        setRenderRangeLocked(false);
        document.getElementById('btn-cancel-reviewed-render').classList.add('hidden');
        if (status.status === 'COMPLETED' && status.output_file) {
          document.getElementById('modal-rerender').classList.add('hidden');
          openVideoPlayer(status.output_file, `审核成片 (${date})`);
        }
      }
    } catch (err) {
      setElementText('rerender-summary', `状态读取失败：${err.message}；任务状态未知，继续查询。`);
    } finally { reviewedRenderPolling = false; }
  };
  reviewedRenderTimer = setInterval(poll, 3000);
  poll();
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



function escapeHTML(value) { return String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char])); }


async function openDeliveryComparison() {
  const item = state.filteredItems[state.selectedIndex];
  if (!item) return showToast('请先选择审核片段', 'info');
  const modal = document.getElementById('modal-delivery-comparison');
  modal.classList.remove('hidden');
  setElementText('delivery-comparison-title', `#${item.id} · 已渲染成片版本`);
  setElementText('delivery-comparison-status', '正在核对交付映射与媒体身份...');
  const body = document.getElementById('delivery-comparison-versions'); body.replaceChildren();
  try {
    const response = await fetch(`/api/delivery_comparison?segment_id=${item.id}`);
    const result = await response.json();
    if (!response.ok || result.error) throw Error(result.error || '交付映射不可用');
    if (state.filteredItems[state.selectedIndex]?.id !== item.id || modal.classList.contains('hidden')) return;
    setElementText('delivery-comparison-status', `${result.status === 'AVAILABLE' ? '' : '当前无可定位交付。'}${result.notice} ${result.problems.join('；')}`);
    for (const version of result.versions) {
      const section = document.createElement('section');
      const heading = document.createElement('p');
      heading.innerText = `${version.role === 'current' ? '当前交付' : '先前交付'} · ${version.generated_at || '时间未记录'} · 版本${version.version_id.slice(0,12)}`;
      section.appendChild(heading);
      version.intervals.forEach((interval,index) => {
        const button = document.createElement('button'); button.className = 'delivery-interval';
        button.innerText = `播放${index+1} · ${interval.state} · 帧${interval.start_frame}–${interval.end_frame} · ${interval.start.toFixed(3)}–${interval.end.toFixed(3)}s`;
        button.addEventListener('click', () => openVideoPlayer(version.filepath, `${heading.innerText} · #${item.id}`, interval,
          `/api/delivery_stream?segment_id=${item.id}&version_id=${version.version_id}`));
        section.appendChild(button);
      });
      body.appendChild(section);
    }
  } catch (error) { setElementText('delivery-comparison-status', error.message); }
}


async function indexCurrentVisualSource() {
  const item = state.filteredItems[state.selectedIndex];
  const id = item?.id || state.visualReference;
  if (!id) return showToast('请先选择参考片段','info');
  const button = document.getElementById('btn-index-visual-source');
  if (button.disabled) return;
  button.disabled = true;
  showToast('正在为当前原片构建视觉索引，请等待；不修改人工标签。','info');
  try {
    const response = await fetch('/api/index_visual_source',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({segment_id:id})});
    const result = await response.json();
    if (!response.ok || !result.success) throw Error(result.error || '索引构建未确认');
    showToast(`当前原片 ${result.indexed}/${result.segments} 条获得有效画面索引`,'success');
    if (state.filteredItems[state.selectedIndex]?.id === id || (!state.filteredItems.length && state.visualReference === id)) {
      state.visualReference = id;
      switchCategory('visual');
    }
  } catch (error) { showToast(`视觉索引不可用：${error.message}`,'error'); }
  finally { button.disabled = false; }
}


// Viewing editions use frozen intervals, never mutable segment IDs or training writes.
(() => {
  const el = id => document.getElementById(`watch-${id}`);
  if (!document.getElementById || !el('date')?.addEventListener) return;
  let preview = null, kind = 'review', offset = 0, total = 0, serial = 0, timer = null;
  const scope = () => ({date: el('date').value.trim(), cam_index: Number(el('camera').value)});
  const query = (extra={}) => new URLSearchParams({...scope(), export_id: el('exports').value, ...extra});
  const status = value => { el('status').textContent = value; };
  async function api(path, params, post=false) {
    const r = await fetch(`/api/watch_${path}${post ? '' : '?'+params}`, post ? {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(params)
    } : {});
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  }
  const safe = fn => async () => { try { await fn(); } catch(e) { status(e.message); } };
  function invalidate() {
    serial++; preview = null; el('start').disabled = true;
    el('exports').replaceChildren(); el('list').replaceChildren(); el('download').removeAttribute('href');
    if (timer) clearTimeout(timer); timer = null;
    status('日期或机位已变化，请重新预览或查找版本。');
  }
  ['date','camera'].forEach(id => el(id).addEventListener('input', invalidate));
  function downloadLink() { el('download').href = `/api/watch_stream?${query({kind:'daily'})}`; }
  async function catalog() {
    const request = ++serial, q = query();
    const d = await api('catalog', q);
    if (request !== serial) return;
    el('exports').replaceChildren();
    d.exports.forEach(id => { const o = document.createElement('option'); o.value=id; o.textContent=id.slice(0,16); el('exports').append(o); });
    status(`已完成版本 ${d.exports.length} 个`); if (d.exports.length) downloadLink();
  }
  el('load').onclick = safe(catalog);
  el('exports').onchange = () => { offset=0; el('list').replaceChildren(); downloadLink(); };
  el('preview').onclick = safe(async () => {
    const request = ++serial, current = scope(); preview=null; el('start').disabled=true;
    const p = await api('preview', new URLSearchParams(current));
    if (request !== serial) return;
    preview = {...current, expected_snapshot:p.snapshot};
    el('start').disabled = p.status !== 'AVAILABLE';
    status(`${p.status} · ${p.duration_s || p.required_seconds || 0}秒 · 日常${p.counts?.daily || 0}段 · 完整争议库存${p.review_total || 0}条 · 优先项${p.required_seconds}秒。`);
  });
  async function poll(request, requestSerial) {
    const s = await api('status', new URLSearchParams({...request, export_id:request.expected_snapshot}));
    if (requestSerial !== serial) return;
    status(`${s.status} · ${s.completed_segments || 0}/${s.total_segments || 0}段 ${s.error || ''}`);
    if (s.worker_alive) timer=setTimeout(() => safe(() => poll(request, requestSerial))(), 1500);
    else if (s.status === 'COMPLETED') await catalog();
  }
  el('start').onclick = safe(async () => {
    if (!preview) throw new Error('请先预览');
    const request={...preview}, requestSerial=serial; el('start').disabled=true;
    const s=await api('start',request,true); status(s.status); await poll(request,requestSerial);
  });
  el('cancel').onclick = safe(async () => {
    if (!preview) throw new Error('没有当前任务');
    const r=await api('cancel',preview,true); status(r.cancel_requested ? '正在取消，等待工作者确认退出' : '任务已退出');
  });
  async function play(mediaKind, itemId=null) {
    const extra={kind:mediaKind}; if(itemId) extra.item_id=itemId;
    const q=query(extra), result=await api('playback',q);
    const stream=`/api/watch_stream?${q}`;
    if (!result.intervals.length) openVideoPlayer('',mediaKind==='daily'?'日常版':'完整版本',null,stream);
    else result.intervals.forEach((interval,index) => {
      if (index === 0) openVideoPlayer('',mediaKind==='source'?'原片正常速度与原声':'完整版本（含上下文）',interval,stream);
    });
    if (result.normal_speed) {
      document.getElementById('vlog-video-el').playbackRate = 1.0;
      document.getElementById('player-interval-hint').textContent = '原片文件内区间，正常速度 1.0×，含原声；请打开音量。未听审声音仍为争议。';
    }
    // Every intersecting full mapping interval remains selectable, including disjoint intervals.
    if (result.intervals.length > 1) {
      const links=document.createElement('div');
      result.intervals.forEach((interval,index) => {
        const b=document.createElement('button'); b.textContent=`完整相交段 ${index+1}`;
        b.onclick=()=>openVideoPlayer('', '完整相交段（含上下文）', interval, stream); links.append(b);
      });
      el('list').prepend(links);
    }
  }
  el('daily').onclick=safe(()=>play('daily')); el('full').onclick=safe(()=>play('full'));
  async function listing() {
    const request=++serial, q=query({kind,offset,limit:50});
    const result=await api('list',q); if(request!==serial) return;
    total=result.total; el('list').replaceChildren(); el('count').textContent=`完整库存 ${total} 条；当前 ${offset+1}–${Math.min(total,offset+50)}（只读冻结区间）`;
    el('prev').disabled=offset===0; el('next').disabled=offset+50>=total;
    for(const item of result.items) {
      const row=document.createElement('div'), text=document.createElement('p');
      const id=item.review_id || item.item_id;
      text.textContent=`${id.slice(0,22)} · ${item.start.toFixed(2)}–${item.end.toFixed(2)}秒 · ${item.disposition || '争议'} · ${(item.reasons || item.dispute_reasons || []).join(', ')} · ${item.source_file}`;
      row.append(text);
      for(const [mode,label] of [['full','看完整相交段（含上下文）'],['source','看原片并听声音（身份验证）']]) {
        const b=document.createElement('button'); b.textContent=label; b.onclick=safe(()=>play(mode,id)); row.append(b);
      }
      el('list').append(row);
    }
  }
  el('review').onclick=safe(async()=>{kind='review';offset=0;await listing();});
  el('selection').onclick=safe(async()=>{kind='selection';offset=0;await listing();});
  el('prev').onclick=safe(async()=>{offset=Math.max(0,offset-50);await listing();});
  el('next').onclick=safe(async()=>{if(offset+50<total){offset+=50;await listing();}});
})();
